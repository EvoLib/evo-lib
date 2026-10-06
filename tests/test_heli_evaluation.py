import pytest

from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.interfaces.enums import EvolutionStrategy
from evolib.operators.heli import run_heli
from evolib.operators.reproduction import generate_cloned_offspring


def test_heli_counts_only_incubation_offspring() -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    pop = Pop(
        config_path="./tests/configs/heli_evaluation.yaml",
        fitness_function=fitness,
    )
    pop.generation_num = 1

    offspring = generate_cloned_offspring(
        pop.indivs,
        1,
        current_gen=pop.generation_num,
    )
    offspring[0].para._has_structural_change = True

    heli_evaluations = run_heli(pop, offspring)

    expected_heli = pop.heli_generations * pop.heli_offspring_per_seed

    assert pop.fitness_evaluations_total == 1
    assert heli_evaluations == expected_heli
    assert len(evaluated_ids) == 1 + expected_heli
    assert len(offspring) == 1
    assert offspring[0].fitness is not None


def test_heli_total_budget_matches_actual_fitness_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    def mark_structural(_pop: Pop, offspring: list[Indiv]) -> None:
        for indiv in offspring:
            indiv.para._has_structural_change = True

    monkeypatch.setattr(
        "evolib.operators.strategy.mutate_offspring",
        mark_structural,
    )

    pop = Pop(
        config_path="./tests/configs/heli_evaluation.yaml",
        fitness_function=fitness,
    )

    pop.run_one_generation(strategy=EvolutionStrategy.MU_PLUS_LAMBDA)

    assert pop.fitness_evaluations_total == pop.mu + pop.lambda_
    assert pop.heli_fitness_evaluations_gen == 0
    assert pop.heli_fitness_evaluations_total == 0
    assert len(evaluated_ids) == pop.fitness_evaluations_total

    pop.run_one_generation(strategy=EvolutionStrategy.MU_PLUS_LAMBDA)

    expected_base_total = pop.mu + 2 * pop.lambda_
    expected_heli = pop.lambda_ * pop.heli_generations * pop.heli_offspring_per_seed

    assert pop.fitness_evaluations_total == expected_base_total
    assert pop.heli_fitness_evaluations_gen == expected_heli
    assert pop.heli_fitness_evaluations_total == expected_heli
    assert len(evaluated_ids) == expected_base_total + expected_heli
    assert len(evaluated_ids) == (
        pop.fitness_evaluations_total + pop.heli_fitness_evaluations_total
    )
