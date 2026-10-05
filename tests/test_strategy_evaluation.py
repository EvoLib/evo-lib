from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.interfaces.enums import EvolutionStrategy


def test_mu_plus_lambda_evaluates_only_unevaluated_individuals() -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    pop = Pop(
        config_path="./tests/configs/population.yaml",
        fitness_function=fitness,
    )

    pop.run_one_generation(strategy=EvolutionStrategy.MU_PLUS_LAMBDA)

    assert len(evaluated_ids) == pop.mu + pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + pop.lambda_

    pop.run_one_generation(strategy=EvolutionStrategy.MU_PLUS_LAMBDA)

    assert len(evaluated_ids) == pop.mu + 2 * pop.lambda_


def test_mu_comma_lambda_reuses_valid_elite_fitness() -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    pop = Pop(
        config_path="./tests/configs/population.yaml",
        fitness_function=fitness,
    )
    pop.num_elites = 1

    pop.run_one_generation(strategy=EvolutionStrategy.MU_COMMA_LAMBDA)

    assert len(evaluated_ids) == pop.mu + pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + pop.lambda_

    pop.run_one_generation(strategy=EvolutionStrategy.MU_COMMA_LAMBDA)

    assert len(evaluated_ids) == pop.mu + 2 * pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + 2 * pop.lambda_


def test_steady_state_evaluates_parents_before_selection() -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    def select_first(pop: Pop, num_parents: int) -> list[Indiv]:
        return [pop.indivs[0].copy() for _ in range(num_parents)]

    pop = Pop(
        config_path="./tests/configs/population.yaml",
        fitness_function=fitness,
    )
    pop.offspring_pool_size = 1
    pop.selection_fn = select_first

    pop.run_one_generation(strategy=EvolutionStrategy.STEADY_STATE)

    assert len(evaluated_ids) == pop.mu + pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + pop.lambda_

    pop.run_one_generation(strategy=EvolutionStrategy.STEADY_STATE)

    assert len(evaluated_ids) == pop.mu + 2 * pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + 2 * pop.lambda_


def test_flexible_reuses_valid_parent_fitness() -> None:
    evaluated_ids: list[str] = []

    def fitness(indiv: Indiv) -> None:
        evaluated_ids.append(indiv.id)
        indiv.fitness = 0.0

    def select_first(pop: Pop, num_parents: int) -> list[Indiv]:
        return [pop.indivs[0].copy() for _ in range(num_parents)]

    def replace_with_offspring(pop: Pop, offspring: list[Indiv]) -> None:
        pop.indivs = offspring[: pop.mu]

    pop = Pop(
        config_path="./tests/configs/population.yaml",
        fitness_function=fitness,
    )
    pop.offspring_pool_size = pop.mu
    pop.selection_fn = select_first
    pop._replacement_fn = replace_with_offspring

    pop.run_one_generation(strategy=EvolutionStrategy.FLEXIBLE)

    assert len(evaluated_ids) == pop.mu + pop.lambda_
    assert pop.fitness_evaluations_total == pop.mu + pop.lambda_

    pop.run_one_generation(strategy=EvolutionStrategy.FLEXIBLE)
