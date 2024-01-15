"""Normalizer is a simple postprocessor that normalizes the values."""
from __future__ import annotations

# Standard Imports
from typing import Any
import operator

# Third-Party Imports
import click

# Perun Imports
from perun.logic import runner
from perun.profile.factory import pass_profile, Profile
from perun.utils.structs import PostprocessStatus


def get_resource_type(resource: dict[str, Any]) -> str:
    """Checks if the resource has defined type and returns empty type otherwise.

    Checks if there is 'type' defined inside the resource, and if so then returns
    the type. Otherwise, it returns empty string as a type.

    :param dict resource: dictionary representing the resource
    :returns str: type of the resource ('' if there is none type)
    """
    return resource.get("type", "")


def normalize_resources(resources: list[dict[str, Any]]) -> None:
    """Normalize the global and snapshot resources according to the maximal values.

    Computes the maximal values per each type inside the snapshot of the resource,
    and then normalizes the values to the interval <0,1>.

    :param list resources: list of resources
    """
    # First compute maxima per each type
    maximum_per_type: dict[str, int] = {}
    for resource in resources:
        resource_type = get_resource_type(resource)
        type_maximum = maximum_per_type.get(resource_type, None)
        if not type_maximum or type_maximum < resource["amount"]:
            maximum_per_type[resource_type] = resource["amount"]

    # Now normalize the values inside the profile
    for resource in resources:
        resource_type = get_resource_type(resource)
        maximum_for_resource_type = maximum_per_type[resource_type]
        resource["amount"] = (
            resource["amount"] / maximum_for_resource_type
            if maximum_for_resource_type != 0.0
            else 1.0
        )


def postprocess(profile: Profile, **_: Any) -> tuple[PostprocessStatus, str, dict[str, Any]]:
    """
    :param Profile profile: json-like profile that will be preprocessed by normalizer
    """
    resources = list(map(operator.itemgetter(1), profile.all_resources()))
    normalize_resources(resources)
    profile.update_resources(resources, clear_existing_resources=True)

    return PostprocessStatus.OK, "", {"profile": profile}


@click.command()
@pass_profile
def normalizer(profile: Profile) -> None:
    """Normalizes performance profile into flat interval.

    \b
      * **Limitations**: `none`
      * **Dependencies**: `none`

    Normalizer is a postprocessor, which iterates through the snapshots
    and normalizes the resources of same type to interval ``(0, 1)``, where
    ``1`` corresponds to the maximal value of the given type.

    Consider the following list of resources for one snapshot generated by
    :ref:`collectors-time`:

    .. code-block:: json

        \b
        [
            {
                'amount': 0.59,
                'uid': 'sys'
            }, {
                'amount': 0.32,
                'uid': 'user'
            }, {
                'amount': 2.32,
                'uid': 'real'
            }
        ]

    Normalizer yields the following set of resources:

    .. code-block:: json

        \b
        [
            {
                'amount': 0.2543103448275862,
                'uid': 'sys'
            }, {
                'amount': 0.13793103448275865,
                'uid': 'user'
            }, {
                'amount': 1.0,
                'uid': 'real'
            }
        ]

    Refer to :ref:`postprocessors-normalizer` for more thorough description and
    examples of `normalizer` postprocessor.
    """
    runner.run_postprocessor_on_profile(profile, "normalizer", {})
