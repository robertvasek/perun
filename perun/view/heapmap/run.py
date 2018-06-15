"""Heap map visualization of the profiles."""

from copy import deepcopy

import click

import perun.profile.convert as heap_representation
import perun.view.heapmap.heap_map as hm
from perun.utils.helpers import pass_profile

__author__ = 'Radim Podola'


def _call_heap(profile):
    """ Call interactive heap map visualization

    :param dict profile: memory profile with records
    """
    heap_map = heap_representation.to_heap_map_format(deepcopy(profile))
    heat_map = heap_representation.to_heat_map_format(profile)
    hm.heap_map(heap_map, heat_map)


@click.command()
@pass_profile
def heapmap(profile, **_):
    """Shows interactive map of memory allocations to concrete memories for
    each function.

    \b
      * **Limitations**: `memory` profiles generated by
        :ref:`collectors-memory`.
      * **Interpretation style**: textual
      * **Visualization backend**: ncurses

    Heap map shows the underlying memory map, and links the concrete
    allocations to allocated addresses for each snapshot. The map is
    interactive, one can either play the full animation of the allocations
    through snapshots or move and explore the details of the map.

    Moreover, the heap map contains `heat map` mode, which accumulates the
    allocations into the heat representation---the hotter the colour displayed
    at given memory cell, the more time it was allocated there.

    The heap map aims at showing the fragmentation of the memory and possible
    differences between different allocation strategies. On the other hand, the
    heat mode aims at showing the bottlenecks of allocations.

    Refer to :ref:`views-heapmap` for more thorough description and example of
    `heapmap` interpretation possibilities.
    """
    _call_heap(profile)
