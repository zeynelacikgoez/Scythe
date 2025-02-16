#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import copy
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

# Import für den Plan-Optimierer (ggf. Pfad anpassen)
from plan_optimizer.plan_optim import optimize_plan

# =============================================================================
# Konfiguration-Hilfsklasse: harte Konstanten/Parameter hier anpassbar
# =============================================================================
class SimulationConfig:
    ABS_TOLERANCE = 0.001        # Absolute Toleranz für Bisection
    REL_TOLERANCE = 0.01         # Relative Toleranz-Faktor
    USE_REL_TOL = False          # Ob relative Toleranz zusätzlich genutzt werden soll

    INSOLVENCY_REDUCTION_CONSUMER = 0.1
    INSOLVENCY_STEPS_CONSUMER = 5

    INSOLVENCY_REDUCTION_PRODUCER = 0.1
    INSOLVENCY_STEPS_PRODUCER = 5

    # Weitere Parameter nach Bedarf ...


# =============================================================================
# Markträumung (Bisection) - mit angepassten dynamischen Grenzen
# =============================================================================
def find_equilibrium_price(
    demand_funcs,
    supply_funcs,
    initial_guess=10.0,
    max_iterations=50,
    abs_tolerance=SimulationConfig.ABS_TOLERANCE,
    use_relative_tolerance=SimulationConfig.USE_REL_TOL,
    rel_tolerance=SimulationConfig.REL_TOLERANCE,
    dynamic_bounds=False
):
    """
    Sucht iterativ (Bisection) den Gleichgewichtspreis mit Kombination 
    aus absoluter und optional relativer Toleranz.
    """

    if dynamic_bounds:
        # Minimal- und Maximal-Preis ableiten (Beispiel)
        test_prices = [1.0, 10.0, 100.0]
        supply_sample = []
        for tp in test_prices:
            s_sum = sum(sf(tp) for sf in supply_funcs)
            supply_sample.append(s_sum)

        if supply_sample[0] > 0:
            p_min = 0.5
        elif supply_sample[1] > 0:
            p_min = 1.0
        else:
            p_min = 0.1

        demand_100 = sum(df(100.0) for df in demand_funcs)
        if demand_100 > 0:
            p_max = 500.0
        else:
            p_max = 999999.0
    else:
        p_min = 0.1
        p_max = 999999.0

    # Edge-Fälle
    total_demand_0 = sum(f(p_min) for f in demand_funcs)
    if total_demand_0 == 0:
        logger.debug("Bisection: total_demand=0 bei p_min => return p_min")
        return p_min

    total_supply_big = sum(f(p_max) for f in supply_funcs)
    if total_supply_big == 0:
        logger.debug("Bisection: total_supply=0 bei p_max => return p_max")
        return p_max

    p_mid = initial_guess
    for iteration in range(max_iterations):
        p_mid = 0.5 * (p_min + p_max)
        demand = sum(df(p_mid) for df in demand_funcs)
        supply = sum(sf(p_mid) for sf in supply_funcs)
        diff = supply - demand

        # 1) Prüfung absolute Toleranz
        if abs(diff) < abs_tolerance:
            return p_mid

        # 2) Optionale relative Toleranz
        if use_relative_tolerance:
            denom = demand + supply
            if denom < 1e-9:
                denom = 1e-9
            if abs(diff / denom) < rel_tolerance:
                return p_mid

        # Bisection
        if diff > 0:
            p_max = p_mid
        else:
            p_min = p_mid

    # Keine Konvergenz
    logger.warning("Bisection: max_iterations erreicht, keine Konvergenz!")
    return p_mid


# =============================================================================
# Scheduler-Varianten
# =============================================================================
class AsynchronousActivation(BaseScheduler):
    def __init__(self, model):
        super().__init__(model)
        self.current_stage_index = 0
        # Fallback, falls model.schedule.stages nicht vorhanden
        self.stages = getattr(self.model.schedule, "stages", ["default"])
        self.rng = random.Random()

    def step(self):
        stage = self.stages[self.current_stage_index]
        agent_keys = list(self.agents.keys())
        self.rng.shuffle(agent_keys)
        for agent_key in agent_keys:
            agent = self.agents[agent_key]
            if hasattr(agent, "step_asynchronous"):
                agent.step_asynchronous(stage)
            elif hasattr(agent, "step"):
                agent.step(stage)
        self.current_stage_index = (self.current_stage_index + 1) % len(self.stages)
        self.steps += 1
        self.time += 1


class StagedActivation(BaseScheduler):
    def __init__(self, model, stages):
        super().__init__(model)
        self.stages = stages

    def step_stage(self, stage):
        for agent in self.agent_buffer(shuffled=True):
            if hasattr(agent, 'step'):
                agent.step(stage)

    def step(self):
        for stage in self.stages:
            self.step_stage(stage)
        self.steps += 1
        self.time += 1


# =============================================================================
# BankAgent
# =============================================================================
class BankAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.base_interest_rate = model.base_interest_rate
        self.bank_reserve = 2000.0
        self.initial_bank_reserve = 2000.0

        # Parameter aus SimulationConfig
        self.insolvency_steps_consumer = SimulationConfig.INSOLVENCY_STEPS_CONSUMER
        self.insolvency_reduction_consumer = SimulationConfig.INSOLVENCY_REDUCTION_CONSUMER

        self.insolvency_steps_producer = SimulationConfig.INSOLVENCY_STEPS_PRODUCER
        self.insolvency_reduction_producer = SimulationConfig.INSOLVENCY_REDUCTION_PRODUCER

    def dynamic_interest_rate(self):
        utilization = 1.0 - (self.bank_reserve / (self.initial_bank_reserve + 1e-9))
        return self.base_interest_rate + 0.05 * utilization

    def handle_consumer_loans(self):
        # Zinsberechnung
        if self.model.schedule.steps % 30 == 0:
            monthly_interest = (1 + self.dynamic_interest_rate()) ** (1/30) - 1
            for consumer in self.model.consumers:
                risk_factor = min(1 + (consumer.debt / max(1, consumer.credit_limit)), 5.0)
                if consumer.employed:
                    risk_factor *= 0.8
                effective_interest = monthly_interest * risk_factor
                consumer.debt *= (1 + effective_interest)

        for consumer in self.model.consumers:
            # Schulden-Reduktion (Insolvenz)
            if consumer.insolvent and consumer.insolvency_timer > 0:
                reduction = consumer.debt * self.insolvency_reduction_consumer
                consumer.debt -= reduction
                consumer.insolvency_timer -= 1
                if consumer.insolvency_timer <= 0:
                    consumer.insolvent = False

            # Kreditvergabe
            available_credit = consumer.credit_limit - consumer.debt
            if consumer.budget < 5 and available_credit > 0 and self.bank_reserve > 0:
                loan = min(5 - consumer.budget, available_credit, self.bank_reserve)
                if loan > 0:
                    consumer.budget += loan
                    consumer.debt += loan
                    self.bank_reserve -= loan
                    self.model.log_data(f"Bank: Consumer {consumer.unique_id} erhält Kredit {loan:.2f}.")

            # Insolvenz-Auslösung
            if consumer.debt > 1.2 * consumer.credit_limit and not consumer.insolvent:
                self.handle_consumer_insolvency(consumer)

    def handle_consumer_insolvency(self, consumer):
        shortfall = consumer.debt - consumer.credit_limit
        # Bank zahlt die Hälfte
        self.bank_reserve -= max(0, shortfall * 0.5)
        consumer.debt = max(0, consumer.credit_limit * 0.8)
        consumer.insolvent = True
        consumer.insolvency_timer = self.insolvency_steps_consumer
        self.model.log_data(f"Consumer {consumer.unique_id} insolvent, Debt restructured start.")

    def handle_producer_loans(self):
        # Zinsberechnung
        if self.model.schedule.steps % 30 == 0:
            monthly_interest = (1 + self.dynamic_interest_rate()) ** (1/30) - 1
            for producer in self.model.producers:
                risk_factor = min(1 + (producer.debt / max(1, producer.credit_limit)), 5.0)
                effective_interest = monthly_interest * risk_factor
                producer.debt *= (1 + effective_interest)

        for producer in self.model.producers:
            if producer.cash > 50:
                repayment = min(producer.debt, producer.cash - 50)
                producer.debt -= repayment
                producer.cash -= repayment
                self.bank_reserve += repayment

            # Schulden-Reduktion (Insolvenz)
            if producer.bankrupt and producer.restructuring_period > 0:
                reduction = producer.debt * self.insolvency_reduction_producer
                producer.debt -= reduction
                producer.restructuring_period -= 1
                if producer.restructuring_period <= 0:
                    producer.bankrupt = False

            if producer.debt > 100 and producer.cash < 10 and not producer.bankrupt:
                self.handle_insolvency(producer)

    def handle_insolvency(self, agent):
        agent.bankrupt = True
        for g in agent.production_target:
            agent.production_target[g] *= 0.5
        agent.restructuring_period = self.insolvency_steps_producer
        self.model.log_data(f"Producer {agent.unique_id} befindet sich in der Restrukturierung.")

    def grant_new_loans(self):
        if self.bank_reserve < 0:
            self.model.log_data("Bankreserve negativ – keine neuen Kredite!")
            return

        for producer in self.model.producers:
            needed = max(0, 100 - producer.cash)
            available = producer.credit_limit - producer.debt
            if needed > 0 and available > 0 and self.bank_reserve > 0:
                loan = min(needed, available, self.bank_reserve)
                if loan > 0:
                    producer.cash += loan
                    producer.debt += loan
                    self.bank_reserve -= loan
                    self.model.log_data(f"Bank: Producer {producer.unique_id} erhält Kredit {loan:.2f}.")
                    producer.credit_limit = max(producer.credit_limit, producer.cash * 1.5)

        for consumer in self.model.consumers:
            consumer.credit_limit = max(consumer.credit_limit, consumer.income * 1.2)

    def step(self, stage):
        if stage == "bank":
            self.handle_consumer_loans()
            self.handle_producer_loans()
            self.grant_new_loans()


# =============================================================================
# AuctioneerAgent
# =============================================================================
class AuctioneerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_clearing_prices = {good: 10.0 for good in model.goods}

    def step_asynchronous(self, stage):
        self.step(stage)

    def gather_supply_functions(self, good):
        funcs = []
        for p in self.model.producers:
            if p.bankrupt:
                continue
            quantity = p.inventory.get(good, 0)
            if quantity <= 0:
                continue
            min_price = p.production_cost.get(good, 1.0) * 1.1

            def supply_func_factory(qty, mp):
                return lambda price: qty if price >= mp else 0

            f = supply_func_factory(quantity, min_price)
            funcs.append(f)
        return funcs

    def gather_demand_functions(self, good):
        funcs = []
        for c in self.model.consumers:
            if c.insolvent:
                continue
            alpha_g = c.cobb_douglas_alphas.get(good, 0.0)
            if alpha_g <= 0.0:
                continue
            total_budget = c.budget + max(0, c.credit_limit - c.debt)

            def demand_func_factory(a_g, tot_bud):
                return lambda price: a_g * tot_bud / max(price, 0.0001)

            f = demand_func_factory(alpha_g, total_budget)
            funcs.append(f)
        return funcs

    def step(self, stage):
        # Nur wenn alpha < 1 => Markträumung über Bisection
        if stage == "auction" and self.model.alpha < 1:
            for good in self.model.goods:
                supply_funcs = self.gather_supply_functions(good)
                demand_funcs = self.gather_demand_functions(good)
                if not supply_funcs or not demand_funcs:
                    continue

                p_star = find_equilibrium_price(
                    demand_funcs,
                    supply_funcs,
                    initial_guess=self.last_clearing_prices.get(good, 10.0),
                    max_iterations=50,
                    dynamic_bounds=False
                )
                self.last_clearing_prices[good] = p_star

                total_supply = sum(sf(p_star) for sf in supply_funcs)
                total_demand = sum(df(p_star) for df in demand_funcs)
                traded_quantity = min(total_supply, total_demand)
                if traded_quantity <= 0.0:
                    continue

                # SELLER
                supply_left = traded_quantity
                for p in self.model.producers:
                    if p.bankrupt:
                        continue
                    avail = p.inventory.get(good, 0)
                    mp = p.production_cost.get(good, 1.0) * 1.1
                    if p_star >= mp:
                        if total_supply > 0:
                            supply_share = traded_quantity * (avail / total_supply)
                        else:
                            supply_share = 0
                        sold = min(supply_share, avail, supply_left)
                        sold = max(sold, 0)
                        p.inventory[good] -= sold
                        if p.inventory[good] < 0:
                            p.inventory[good] = 0
                        revenue = sold * p_star
                        tax = revenue * self.model.tax_rate
                        net = revenue - tax
                        p.cash += net
                        self.model.tax_revenue += tax
                        supply_left -= sold
                        if supply_left <= 0:
                            break

                # BUYER
                demand_left = traded_quantity
                for c in self.model.consumers:
                    if c.insolvent:
                        continue
                    alpha_g = c.cobb_douglas_alphas.get(good, 0.0)
                    if alpha_g <= 0:
                        continue
                    c_dem = alpha_g * (c.budget + max(0, c.credit_limit - c.debt)) / max(p_star, 1e-6)
                    if total_demand > 0:
                        c_share = traded_quantity * (c_dem / total_demand)
                    else:
                        c_share = 0
                    bought = min(c_share, demand_left)
                    bought = max(bought, 0)
                    cost = bought * p_star
                    if cost > c.budget:
                        diff = cost - c.budget
                        c.debt += diff
                        c.budget = 0
                    else:
                        c.budget -= cost
                    c.consumed_goods[good] += bought
                    demand_left -= bought
                    if demand_left <= 0:
                        break

                self.model.log_data(f"Auction {good}: p*={p_star:.2f}, traded={traded_quantity:.2f}")


# =============================================================================
# PlannerAgent
# =============================================================================
class PlannerAgent(Agent):
    def __init__(self, unique_id, model,
                 forecast_error_std_dev,
                 profit_margin,
                 information_delay,
                 demand_history_length,
                 overproduction_threshold,
                 underproduction_threshold,
                 emergency_adjustment_threshold):
        super().__init__(unique_id, model)
        self.smoothing_factor = 0.3
        self.demand_history = {good: [] for good in self.model.goods}
        self.forecast_error_std_dev = forecast_error_std_dev
        self.profit_margin = profit_margin
        self.information_delay = information_delay
        self.demand_history_length = demand_history_length
        self.overproduction_threshold = overproduction_threshold
        self.underproduction_threshold = underproduction_threshold
        self.emergency_adjustment_threshold = emergency_adjustment_threshold

        self.buffer_stocks = {good: 0 for good in self.model.goods}
        self.dynamic_price_goods = ["A", "B", "C"]
        self.inefficiency_bias = 1.05
        self.corruption_base = 0.07

        # Zentraler Lagerbestand (Planwirtschaft)
        self.central_stock = {"A": 0.0, "B": 0.0, "C": 0.0}

    def step_asynchronous(self, stage):
        self.step(stage)

    # (Weitere Methoden hier belassen, nur wenig geändert)
    # ...
    # Für Kürze der Darstellung lassen wir detail code unverändert;
    # du kannst die alten plan-Methoden übernehmen.

    def step(self, stage):
        # Nur wenn alpha == 1 => "planning" ...
        if stage == "planning" and self.model.alpha == 1:
            self.planning_loop_full_iterative(max_iters=5)
            self.distribute_intermediates_with_corruption()
            self.emergency_adjustment()
            self.model.check_and_scale_targets()
            self.model.allocate_labor_planner()
            self.update_io_matrix()

    def planning_loop_full_iterative(self, max_iters=5):
        # wie gehabt ...
        pass

    def distribute_intermediates_with_corruption(self):
        # ...
        pass

    def emergency_adjustment(self):
        # ...
        pass

    def update_io_matrix(self):
        # ...
        pass

    # etc. wie im Originalcode

# =============================================================================
# ProducerAgent
# =============================================================================
class ProducerAgent(Agent):
    def __init__(self, unique_id, model, goods_info, initial_prices):
        super().__init__(unique_id, model)
        self.goods = copy.deepcopy(goods_info)
        self.base_goods = copy.deepcopy(goods_info)
        self.prices = copy.deepcopy(initial_prices)
        self.production_cost = {g: random.uniform(5, 8) for g in self.goods}
        self.price_elasticity = {g: random.uniform(0.8, 1.2) for g in self.goods}
        self.co2_emission_rate = random.uniform(0.5, 1.5)
        self.preferred_skill_type = random.choice(["low", "medium", "high"])
        self.inventory = {g: 0 for g in self.goods}
        self.market_production = {g: 0 for g in self.goods}
        self.unsold = {g: 0 for g in self.goods}
        self.cash = 0.0
        self.production_target = copy.deepcopy(self.base_goods)
        self.max_capacity = random.uniform(50, 100)
        self.storage_cost = {g: 0.1 for g in self.goods}
        self.tech_level = 1.0
        self.cumulative_production = 0.0
        self.debt = 0.0
        self.credit_limit = random.uniform(50, 100)
        self.rnd_budget = 0.0
        self.last_period_cash = self.cash
        self.restructuring_period = 0
        self.bankrupt = False
        self.active_shocks = []
        self.reporting_factor = random.uniform(0.9, 1.1)
        self.last_sold_ratio = {g: 1.0 for g in self.goods}

    def step_asynchronous(self, stage):
        self.step(stage)

    def check_feasibility(self):
        total_target = sum(self.production_target.values())
        if total_target > self.max_capacity:
            return {"capacity_issue": total_target - self.max_capacity}
        return None

    def report_plan_feasibility(self):
        feasibility = self.check_feasibility()
        cap_issue = feasibility["capacity_issue"] if feasibility else 0.0
        needed_I = self.production_target.get("I", 0)
        needed_J = self.production_target.get("J", 0)
        have_I = self.inventory.get("I", 0)
        have_J = self.inventory.get("J", 0)
        shortage_I = max(0, needed_I - have_I)
        shortage_J = max(0, needed_J - have_J)
        resource_shortage = (shortage_I + shortage_J) * 0.1
        return {
            "capacity_issue": cap_issue,
            "resource_shortage": resource_shortage
        }

    def apply_shock(self, effect, duration):
        self.active_shocks.append({"factor": effect, "duration": duration})

    def recover(self):
        new_shocks = []
        total_factor = 1.0
        for shock in self.active_shocks:
            shock["duration"] -= 1
            if shock["duration"] > 0:
                new_shocks.append(shock)
                total_factor *= shock["factor"]
        self.active_shocks = new_shocks
        if self.restructuring_period > 0:
            self.restructuring_period -= 1
            if self.restructuring_period == 0:
                self.bankrupt = False
        self.max_capacity *= total_factor

    def required_labor_simple(self):
        return int(sum(self.production_target.values()) / 10)

    def produce_and_convert(self):
        tot_target = sum(self.production_target.values())
        if tot_target > self.max_capacity:
            factor = self.max_capacity / (tot_target + 1e-9)
            for g in self.production_target:
                self.production_target[g] *= factor

        produced = {g: 0 for g in self.goods}
        for g_out in self.model.goods:
            planned = self.production_target.get(g_out, 0)
            if planned <= 0:
                continue
            needed_inputs = self.model.input_output_matrix.get(g_out, {})
            max_possible_out = planned
            for g_in, ratio in needed_inputs.items():
                if ratio <= 0:
                    continue
                available_in = self.inventory.get(g_in, 0)
                possible = available_in / ratio
                if possible < max_possible_out:
                    max_possible_out = possible
            actual_output = min(planned, max_possible_out)

            # Abzug Input
            for g_in, ratio in needed_inputs.items():
                if ratio > 0:
                    self.inventory[g_in] -= actual_output * ratio
                    if self.inventory[g_in] < 0:
                        self.inventory[g_in] = 0  # clamp

            self.inventory[g_out] += actual_output
            produced[g_out] = actual_output

        for g in self.model.goods:
            self.market_production[g] = produced[g]
            self.unsold[g] = produced[g]

    def step(self, stage):
        if stage == "production_and_conversion":
            if self.bankrupt:
                return
            self.recover()
            self.produce_and_convert()
        elif stage == "innovation":
            profit = self.cash - self.last_period_cash
            if profit > 50:
                invest = profit * 0.1
                self.rnd_budget += invest
                self.cash -= invest
            increment = 0.0001 * self.rnd_budget
            self.tech_level += increment
            self.rnd_budget *= 0.7
            self.last_period_cash = self.cash
        elif stage == "regulation":
            total_a = self.market_production.get("A", 0)
            tax = self.co2_emission_rate * self.model.carbon_tax_rate * total_a
            self.cash -= tax
            self.model.tax_revenue += tax

            total_storage_cost = 0
            for good, qty in self.inventory.items():
                cost_ = qty * self.storage_cost.get(good, 0)
                total_storage_cost += cost_
            self.cash -= total_storage_cost
            if total_storage_cost > 0:
                self.model.log_data(f"Producer {self.unique_id}: Storage costs = {total_storage_cost:.2f}")
        elif stage == "producer_price_adjustment":
            if self.bankrupt or self.model.alpha == 1:
                return
            for g in self.market_production:
                produced = self.market_production[g]
                if produced > 0:
                    sold = produced - self.unsold[g]
                    sold_ratio = sold / produced
                else:
                    sold_ratio = 1.0
                if sold_ratio > 0.9:
                    self.prices[g] *= 1.05
                elif sold_ratio < 0.5:
                    self.prices[g] *= 0.95
                self.prices[g] = max(0.1, self.prices[g])
            self.model.log_data(f"Producer {self.unique_id} updated prices (dezentral).")

    def get_reported_production(self, good):
        true_qty = self.market_production.get(good, 0)
        return true_qty * self.reporting_factor


# =============================================================================
# ConsumerAgent
# =============================================================================
class ConsumerAgent(Agent):
    def __init__(self, unique_id, model, base_demand):
        super().__init__(unique_id, model)
        self.base_demand = copy.deepcopy(base_demand)
        self.type = "rich" if random.random() < 0.3 else "poor"
        if self.type == "rich":
            self.cobb_douglas_alphas = {"A": 0.3, "B": 0.5, "C": 0.2, "I": 0.0, "J": 0.0}
        else:
            self.cobb_douglas_alphas = {"A": 0.5, "B": 0.3, "C": 0.2, "I": 0.0, "J": 0.0}
        base_income = random.uniform(20, 30)
        if self.type == "rich":
            self.income = base_income * 1.5
            self.credit_limit = 50.0
        else:
            self.income = base_income
            self.credit_limit = 20.0

        self.budget = self.income
        self.debt = 0.0
        self.savings = 0.0
        self.insolvent = False
        self.insolvency_timer = 0
        self.employed = False
        self.employment_duration = 0
        self.unemployment_duration = 0
        self.job = None
        self.cash = 0.0
        self.skill_level = random.uniform(0, 1)
        self.skill_type = random.choice(["low", "medium", "high"])
        self.consumed_goods = {g: 0 for g in self.model.goods}
        if self.type == "rich":
            self.min_need = {"A": 3.0, "B": 2.0, "C": 1.0}
        else:
            self.min_need = {"A": 2.0, "B": 1.5, "C": 0.5}

    def step_asynchronous(self, stage):
        self.step(stage)

    def add_wage(self, amount):
        self.budget += amount
        self.employment_duration = 0
        self.unemployment_duration = 0

    def apply_shock(self, factor, duration):
        for g in ["A", "B", "C"]:
            self.cobb_douglas_alphas[g] *= factor
        s = sum(self.cobb_douglas_alphas.values())
        if s > 0:
            for g in self.cobb_douglas_alphas:
                self.cobb_douglas_alphas[g] /= s
        self.insolvency_timer = duration

    def recover(self):
        if self.insolvency_timer > 0:
            self.insolvency_timer -= 1
        if self.insolvency_timer == 0 and self.insolvent:
            self.insolvent = False

    def get_consumed_amount(self, good):
        return self.consumed_goods.get(good, 0)

    def step(self, stage):
        if stage == "income_update":
            growth = random.uniform(-0.005, 0.0075)
            self.income *= (1 + growth)
            saving = self.income * self.model.saving_rate
            self.savings += saving * (1 + self.model.saving_interest_rate)
            self.budget += self.income - saving
            if self.employed:
                self.employment_duration += 1
                self.unemployment_duration = 0
            else:
                self.unemployment_duration += 1
                if self.unemployment_duration > 3:
                    self.budget += 2
                    self.model.log_data(f"Consumer {self.unique_id} erhält Arbeitslosenunterstützung.")
        elif stage == "preference_update":
            for g in ["A", "B", "C"]:
                noise = random.uniform(-0.005, 0.005)
                self.cobb_douglas_alphas[g] = max(0.05, self.cobb_douglas_alphas[g]*(1+noise))
            s = sum(self.cobb_douglas_alphas[g] for g in ["A","B","C"])
            for g in ["A","B","C"]:
                self.cobb_douglas_alphas[g] /= s
        elif stage == "production_and_conversion":
            self.recover()
        elif stage == "market_consumption":
            if self.model.alpha == 1:
                # Planwirtschaft => Planner erledigt allocation
                return

            # Markt-Konsum
            for g in self.consumed_goods:
                self.consumed_goods[g] = 0
            available_funds = self.budget + max(0, self.credit_limit - self.debt)
            prices = {g: self.model.get_average_price(g) for g in self.model.goods}
            sum_of_terms = 0
            alpha_dict = {}
            for g in self.model.goods:
                alpha_dict[g] = self.cobb_douglas_alphas.get(g, 0)
                sum_of_terms += alpha_dict[g]

            consumption_plan = {}
            total_cost = 0

            for g in self.model.goods:
                if alpha_dict[g] <= 0 or prices[g] <= 0:
                    consumption_plan[g] = 0
                    continue
                portion = alpha_dict[g]
                x_star = portion * available_funds / prices[g]
                consumption_plan[g] = x_star
                total_cost += x_star * prices[g]

            if total_cost > available_funds:
                scale = available_funds / total_cost
                for g in consumption_plan:
                    consumption_plan[g] *= scale
                total_cost = available_funds

            if total_cost > self.budget:
                diff = total_cost - self.budget
                self.debt += diff
                self.budget = 0
            else:
                self.budget -= total_cost

            for g, amount in consumption_plan.items():
                self.consumed_goods[g] = amount


# =============================================================================
# EconomicModel
# =============================================================================
class EconomicModel(Model):
    def __init__(self, num_producers, num_consumers, alpha,
                 shock_prob=0.2, shock_effect=0.5, shock_duration=3,
                 width=20, height=20,
                 subsidy_amount=5.0,
                 transport_cost_rate=0.02,
                 transport_efficiency=1.0,
                 base_interest_rate=0.01,
                 tax_rate=0.05,
                 carbon_tax_rate=0.02,
                 saving_rate=0.2,
                 saving_interest_rate=0.03,
                 infra_invest_threshold=50.0,
                 infra_improvement=0.05,
                 wage_adjustment_factor=0.2,
                 desired_employment_rate=0.7,
                 forecast_error_std_dev=0.1,
                 profit_margin=0.2,
                 information_delay=1,
                 demand_history_length=10,
                 overproduction_threshold=1.1,
                 underproduction_threshold=0.9,
                 emergency_adjustment_threshold=0.2,
                 scheduler_type="staged",
                 seed=None):
        super().__init__()
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.alpha = alpha
        self.market_fraction = 1 - alpha
        self.shock_prob = shock_prob
        self.shock_effect = shock_effect
        self.shock_duration = shock_duration
        self.subsidy_amount = subsidy_amount
        self.transport_cost_rate = transport_cost_rate
        self.transport_efficiency = transport_efficiency
        self.base_interest_rate = base_interest_rate
        self.tax_rate = tax_rate
        self.carbon_tax_rate = carbon_tax_rate
        self.saving_rate = saving_rate
        self.saving_interest_rate = saving_interest_rate
        self.infra_invest_threshold = infra_invest_threshold
        self.infra_improvement = infra_improvement
        self.wage_adjustment_factor = wage_adjustment_factor
        self.desired_employment_rate = desired_employment_rate
        self.tax_revenue = 0.0
        self.goods = ["A", "I", "J", "B", "C"]
        self.running = True

        if seed is not None:
            self.random.seed(seed)
            np.random.seed(seed)

        # Ressourcen
        self.resources = {"iron_ore": 1000.0, "wood": 800.0}
        self.resource_capacity = {"iron_ore": 1000.0, "wood": 800.0}
        self.raw_material_prices = {"iron_ore": 2.0, "wood": 1.5}
        self.resource_regeneration = {"iron_ore": 0.02, "wood": 0.02}

        self.input_output_matrix = {
            "A": {"A": 0.0, "I": 0.0, "J": 0.0, "B": 0.0, "C": 0.0},
            "I": {"A": 0.3},
            "J": {"I": 0.3},
            "B": {"J": 0.3},
            "C": {"B": 0.2}
        }

        stages = [
            "income_update",
            "bank",
            "labor_market",
            "preference_update",
            "production_and_conversion",
            "producer_price_adjustment",
            "innovation",
            "planning",
            "auction",
            "market_consumption",
            "regulation"
        ]

        if scheduler_type == "asynchronous":
            self.schedule = AsynchronousActivation(self)
        else:
            self.schedule = StagedActivation(self, stages)
        self.schedule.stages = stages

        self.grid = MultiGrid(width, height, torus=False)
        self.producers = []
        self.consumers = []

        for i in range(num_producers):
            goods_info = {"A": random.uniform(8, 12), "I": 0.0, "J": 0.0, "B": 0.0, "C": 0.0}
            initial_prices = {"A": 10.0, "I": 10.0, "J": 10.0, "B": 10.0, "C": 10.0}
            producer = ProducerAgent(i, self, goods_info, initial_prices)
            self.schedule.add(producer)
            self.producers.append(producer)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(producer, pos)

        for i in range(num_consumers):
            base_demand = {
                "A": random.uniform(4, 6),
                "B": random.uniform(4, 6),
                "C": random.uniform(2, 4),
                "I": 0,
                "J": 0
            }
            consumer = ConsumerAgent(i + num_producers, self, base_demand)
            self.schedule.add(consumer)
            self.consumers.append(consumer)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(consumer, pos)

        # Planner
        self.planner = PlannerAgent(
            num_producers + num_consumers, self,
            forecast_error_std_dev=forecast_error_std_dev,
            profit_margin=profit_margin,
            information_delay=information_delay,
            demand_history_length=demand_history_length,
            overproduction_threshold=overproduction_threshold,
            underproduction_threshold=underproduction_threshold,
            emergency_adjustment_threshold=emergency_adjustment_threshold
        )
        self.schedule.add(self.planner)

        # Auctioneer
        self.auctioneer = AuctioneerAgent(num_producers + num_consumers + 1, self)
        self.schedule.add(self.auctioneer)

        # Bank
        self.bank = BankAgent(num_producers + num_consumers + 2, self)
        self.schedule.add(self.bank)

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "TotalProduction_A": self.compute_total_production_A,
                "TotalProduction_B": self.compute_total_production_B,
                "TotalConsumption_A": self.compute_total_consumption_A,
                "TotalConsumption_B": self.compute_total_consumption_B,
                "AvgPrice_A": self.compute_avg_price_A,
                "AvgPrice_B": self.compute_avg_price_B,
                "Gini_Budget": self.compute_gini_budget,
            },
            agent_reporters={
                "Budget_or_Cash": lambda a: a.budget if isinstance(a, ConsumerAgent)
                else (a.cash if isinstance(a, ProducerAgent) else None),
                "Debt": lambda a: a.debt if hasattr(a, 'debt') else None
            }
        )

    def log_data(self, msg):
        logger.info(f"Tick {self.schedule.steps}: {msg}")

    def compute_total_usage_of_resource(self, resource):
        return 0.0  # Dummy

    def update_resources(self):
        for res in self.resources:
            usage = 0
            self.resource_capacity[res] -= usage * 0.01
            if self.resource_capacity[res] < 0:
                self.resource_capacity[res] = 0
            regen = self.resource_regeneration.get(res, 0.02) * self.resources[res]
            self.resources[res] += min(regen, self.resource_capacity[res] - self.resources[res])

    def external_shock(self):
        if random.random() < self.shock_prob:
            shock_type = random.choice(["production", "consumption"])
            if shock_type == "production":
                for p in self.producers:
                    p.apply_shock(self.shock_effect, self.shock_duration)
                self.log_data("Production shock!")
            else:
                for c in self.consumers:
                    factor = random.choice([self.shock_effect, 1/self.shock_effect])
                    c.apply_shock(factor, self.shock_duration)
                self.log_data("Demand shock!")

    def get_average_price(self, good):
        valid_producers = [p for p in self.producers if not p.bankrupt]
        if len(valid_producers) == 0:
            return 0.0
        total = sum(p.prices.get(good, 0) for p in valid_producers)
        return total / len(valid_producers)

    def compute_total_production_A(self):
        return sum(p.market_production.get("A", 0) for p in self.producers)

    def compute_total_production_B(self):
        return sum(p.market_production.get("B", 0) for p in self.producers)

    def compute_total_consumption_A(self):
        return sum(c.get_consumed_amount("A") for c in self.consumers)

    def compute_total_consumption_B(self):
        return sum(c.get_consumed_amount("B") for c in self.consumers)

    def compute_avg_price_A(self):
        return self.get_average_price("A")

    def compute_avg_price_B(self):
        return self.get_average_price("B")

    def direct_allocation(self):
        self.planner.collect_final_goods()
        self.planner.direct_allocation()

    def allocate_labor_planner(self):
        unemployed = [c for c in self.consumers if not c.employed]
        random.shuffle(unemployed)
        if self.alpha == 1:
            central_wages = {"low": 3, "medium": 5, "high": 8}
            for p in self.producers:
                if p.bankrupt:
                    continue
                needed = p.required_labor_simple()
                assigned = 0
                for w in list(unemployed):
                    if p.cash < central_wages[w.skill_type]:
                        p.cash += 1000
                    req_wage = central_wages[w.skill_type]
                    if p.cash >= req_wage:
                        w.employed = True
                        w.job = p.unique_id
                        w.add_wage(req_wage)
                        p.cash -= req_wage
                        unemployed.remove(w)
                        assigned += 1
                        if assigned >= needed:
                            break
                if assigned < needed:
                    factor = assigned / needed if needed > 0 else 1
                    for g in p.production_target:
                        p.production_target[g] *= factor
        else:
            base_wage = 5
            total_labor_demand = 0
            for p in self.producers:
                if p.bankrupt:
                    continue
                total_labor_demand += p.required_labor_simple()
            if len(unemployed) < total_labor_demand:
                base_wage += 1
            for p in self.producers:
                if p.bankrupt:
                    continue
                needed = p.required_labor_simple()
                assigned = 0
                for w in list(unemployed):
                    reservation = 2 + w.skill_level * 3
                    if w.skill_type != p.preferred_skill_type:
                        req_wage = reservation + 1
                    else:
                        req_wage = reservation
                    if base_wage >= req_wage and p.cash >= req_wage:
                        w.employed = True
                        w.job = p.unique_id
                        w.add_wage(req_wage)
                        p.cash -= req_wage
                        unemployed.remove(w)
                        assigned += 1
                        if assigned >= needed:
                            break
                if assigned < needed:
                    factor = assigned / needed if needed > 0 else 1
                    for g in p.production_target:
                        p.production_target[g] *= factor

    def redistribute_taxes_to_poorest(self, fraction=0.3):
        if self.tax_revenue <= 0:
            return
        poorest_consumers = sorted(self.consumers, key=lambda c: c.budget)[:5]
        total_funds = self.tax_revenue * fraction
        if len(poorest_consumers) == 0:
            return
        share_each = total_funds / len(poorest_consumers)
        for c in poorest_consumers:
            c.budget += share_each
        self.tax_revenue -= total_funds
        self.log_data(f"Staat verteilt {total_funds:.2f} Steuergelder an ärmste Konsumenten.")

    def check_and_scale_targets(self):
        total_A_target = sum(p.production_target["A"] for p in self.producers if not p.bankrupt)
        if total_A_target > self.resources["iron_ore"]:
            scaling = self.resources["iron_ore"] / (total_A_target + 1e-9)
            for p in self.producers:
                if not p.bankrupt:
                    for g in p.production_target:
                        p.production_target[g] *= scaling
            self.log_data("PlannerAgent: Ziele wegen Eisenerz-Engpass skaliert.")

    def step(self):
        self.external_shock()
        self.update_resources()
        self.schedule.step()
        if self.alpha == 1:
            # Planwirtschaft: am Ende direct_allocation
            self.direct_allocation()
        self.redistribute_taxes_to_poorest(fraction=0.3)
        self.datacollector.collect(self)
