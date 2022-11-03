from automatic_prompt_engineer.evaluation import bandits


def test_counting():
    algo = bandits.UCBBanditAlgo(5, 1, c=1)
    algo.update(chosen=[0, 1], scores=[5, 10])
    algo.update(chosen=[0, 1, 2, 3, 4], scores=[1, 1, 1, 1, 1])
    scores = algo.get_scores()
    assert scores[0] == (5 + 1) / 2


def test_ucb():
    algo = bandits.UCBBanditAlgo(5, 1, c=1)
    algo.update(chosen=[0, 1], scores=[5, 10])
    algo.update(chosen=[0, 1, 2, 3], scores=[1, 1, 1, 1])
    chosen = algo.choose(n=3)
    assert len(chosen) == 3
    assert chosen[0] == 4
    assert chosen[1] == 1
