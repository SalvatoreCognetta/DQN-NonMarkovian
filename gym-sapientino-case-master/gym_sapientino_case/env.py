"""Gym interface."""

from pathlib import Path
from typing import Optional, cast

import gym
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import types as sapientino_types
from gym_sapientino.core.actions import ContinuousCommand
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.wrappers import observations
from gym_sapientino.wrappers.gym import SingleAgentWrapper
from logaut import ldl2dfa
from pylogics.parsers import parse_ldl
from pythomata.core import Rendering as RenderingDFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper, TemporalGoalWrapperSynthetic

# One map
_default_map = """\
|           |
|     b     |
|     #     |
|  r  # g   |
|     #     |
|###########|"""


class SapientinoCase(gym.Wrapper):
    """A specific instance of gym sapientino with non-markovian goal."""

    def __init__(
        self,
        conf: SapientinoConfiguration = None,
        reward_ldlf: Optional[str] = None,
        logdir: Optional[str] = None,
    ):
        """Initialize.

        :param conf: an environment configuration; see the
            SapientinoConfiguration class.
        :param reward_ldlf: a LDLf temporal goal formula to generate +1 reward.
        :param logdir: where to save logs.
        """
        # Use defaults if a configuration not given. Best not to rely on this
        if conf is None:
            agent_conf = SapientinoAgentConfiguration(
                initial_position=(2, 2),
                commands=ContinuousCommand,
                angular_speed=30.0,
                acceleration=0.10,
                max_velocity=0.40,
                min_velocity=0.0,
            )

            conf = SapientinoConfiguration(
                agent_configs=(agent_conf,),
                grid_map=_default_map,
                reward_outside_grid=0.0,
                reward_duplicate_beep=0.0,
                reward_per_step=0.0,
            )

        # Instantiate the environment
        env = SapientinoDictSpace(configuration=conf)

        # Choose an observation space
        env = observations.UseFeatures(
            env=env,
            features=[observations.ContinuousFeatures],
        )
        env_with_features = env
        env = SingleAgentWrapper(env)

        # Default temporal goal
        if reward_ldlf is None:
            reward_ldlf = "<!red*; red; !green*; green; !blue*; blue>end"

        # Automaton and composition
        print("> Parsing LDLf")
        dfa = ldl2dfa(parse_ldl(reward_ldlf))
        if logdir is not None:
            RenderingDFA.to_graphviz(cast(RenderingDFA, dfa)).render(
                Path(logdir) / "reward-dfa.dot", format="pdf")
        print("> Parsed")

        env_ = TemporalGoalWrapper(
            env=env,
            temp_goals=[
                TemporalGoal(
                    reward=1.0,
                    automaton=dfa,
                )],
            fluent_extractor=ColorExtractor(env_with_features),
        )

        self.env_synthetic = TemporalGoalWrapperSynthetic(
            env=env,
            automaton=dfa
        )

        # Save
        super().__init__(env_)

    def get_synthetic_env(self) -> TemporalGoalWrapperSynthetic:
        return self.env_synthetic


class ColorExtractor:
    """A fluent extractor for this environment."""

    def __init__(self, env: observations.UseFeatures):
        """Initialize.

        :param env: a wrapped gym sapientino environment with one agent.
        """
        self.env = env

        # Transform color id to colors
        self._int2color = {i: str(c) for c, i in sapientino_types.color2int.items()}

    def __call__(self, observation, action):  # type: ignore
        """Fluent extractor."""
        assert len(self.env.last_dict_observation) == 1
        obs = self.env.last_dict_observation[0]
        color = self._int2color[obs["color"]]
        if (color == "blank" or action is None or
                ContinuousCommand(action) != ContinuousCommand.BEEP):
            return frozenset()
        else:
            return frozenset({color})
