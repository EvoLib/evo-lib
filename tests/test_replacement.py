from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.interfaces.enums import Origin
from evolib.operators.replacement import (
    replace_generational,
    replace_mu_plus_lambda,
    replace_truncation,
)


def _indiv(fitness: float, origin: Origin = Origin.PARENT) -> Indiv:
    indiv = Indiv()
    indiv.fitness = fitness
    indiv.origin = origin
    return indiv


def test_generational_preserves_elite_and_marks_removed() -> None:
    pop = Pop(config_path="./tests/configs/population.yaml", initialize=False)
    pop.num_elites = 1
    pop.generation_num = 3

    elite = _indiv(1.0)
    removed_parent = _indiv(10.0)
    pop.indivs = [elite, removed_parent]

    selected_offspring = _indiv(0.0, Origin.OFFSPRING)
    rejected_offspring = _indiv(0.5, Origin.OFFSPRING)
    offspring = [selected_offspring, rejected_offspring, _indiv(2.0), _indiv(3.0)]

    replace_generational(pop, offspring)

    assert set(pop.indivs) == {elite, selected_offspring}
    assert removed_parent.exit_gen == 3
    assert rejected_offspring.exit_gen == 3


def test_truncation_uses_parents_and_preserves_offspring_origin() -> None:
    pop = Pop(config_path="./tests/configs/population.yaml", initialize=False)
    pop.generation_num = 4

    best_parent = _indiv(1.0)
    removed_parent = _indiv(10.0)
    pop.indivs = [best_parent, removed_parent]

    selected_offspring = _indiv(2.0, Origin.OFFSPRING)
    rejected_offspring = _indiv(3.0, Origin.OFFSPRING)

    replace_truncation(pop, [selected_offspring, rejected_offspring])

    assert pop.indivs == [best_parent, selected_offspring]
    assert selected_offspring.origin == Origin.OFFSPRING
    assert removed_parent.exit_gen == 4
    assert rejected_offspring.exit_gen == 4

    pop.age_indivs()

    assert selected_offspring.origin == Origin.PARENT


def test_mu_plus_lambda_passes_only_offspring_to_truncation() -> None:
    pop = Pop(config_path="./tests/configs/population.yaml", initialize=False)

    best_parent = _indiv(1.0)
    other_parent = _indiv(10.0)
    pop.indivs = [best_parent, other_parent]

    best_offspring = _indiv(2.0, Origin.OFFSPRING)
    other_offspring = _indiv(3.0, Origin.OFFSPRING)

    replace_mu_plus_lambda(pop, [best_offspring, other_offspring])

    assert pop.indivs == [best_parent, best_offspring]
    assert len({indiv.id for indiv in pop.indivs}) == pop.parent_pool_size
