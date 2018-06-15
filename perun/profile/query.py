"""``perun.profile.query`` is a module which specifies interface for issuing
queries over the profiles w.r.t :ref:`profile-spec`.

.. _Pandas library: https://docs.python.org/3.7/library/json.html

Run the following in the Python interpreter to extend the capabilities of
profile to query over profiles, iterate over resources or models, etc.::

    import perun.profile.query

Combined with ``perun.profile.factory``, ``perun.profile.convert`` and e.g.
`Pandas library`_ one can obtain efficient interpreter for executing more
complex queries and statistical tests over the profiles.


"""
# TODO: Consider adding caching to ease some of the computations (in case it is
# time consuming)

import operator
import numbers
import perun.utils.exceptions as exceptions
import perun.utils.helpers as helpers

__author__ = 'Tomas Fiedor'
__coauthored__ = "Jiri Pavela"


def all_resources_of(profile):
    """Generator for iterating through all of the resources contained in the
    performance profile.

    Generator iterates through all of the snapshots, and subsequently yields
    collected resources. For more thorough description of format of resources
    refer to :pkey:`resources`. Resources are not flattened and, thus, can
    contain nested dictionaries (e.g. for `traces` or `uids`).

    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :returns: iterable stream of resources represented as pair ``(int, dict)``
        of snapshot number and the resources w.r.t. the specification of the
        :pkey:`resources`
    :raises AttributeError: when the profile is not in the format as given
        by :ref:`profile-spec`
    :raises KeyError: when the profile misses some expected key, as given
        by :ref:`profile-spec`
    """
    try:
        # Get snapshot resources
        snapshots = profile.get('snapshots', [])
        for snap_no, snapshot in enumerate(snapshots):
            for resource in snapshot['resources']:
                yield snap_no, resource

        # Get global resources
        resources = profile.get('global', {}).get('resources', [])
        for resource in resources:
            yield len(snapshots), resource

    except AttributeError:
        # Element is not dict-like type with get method
        raise exceptions.IncorrectProfileFormatException(
            'profile', "Expected dictionary, got different type.") from None
    except KeyError:
        # Dictionary does not contain specified key
        raise exceptions.IncorrectProfileFormatException(
            'profile', "Missing key in dictionary.") from None


def flattened_values(root_key, root_value):
    """Converts the (root_key, root_value) pair to something that can be added to table.

    Flattens all of the dictionaries to single level and <key>(:<key>)? values, lists are processed
    to comma separated representation and rest is left as it is.

    :param str or int root_key: name (or index) of the processed key, that is going to be flattened
    :param object root_value: value that is flattened
    :returns (key, object): either decimal, string, or something else
    """
    # Dictionary is processed recursively according to the all items that are nested
    if isinstance(root_value, dict):
        nested_values = []
        for key, value in all_items_of(root_value):
            # Add one level of hierarchy with ':'
            nested_values.append((key, value))
            yield str(root_key) + ":" + key, value
        # Additionally return the overall key as joined values of its nested stuff,
        # only if root is not a list (i.e. root key is not int = index)!
        if isinstance(root_key, str):
            nested_values.sort(key=helpers.uid_getter)
            yield root_key, ":".join(map(str, map(operator.itemgetter(1), nested_values)))
    # Lists are merged as comma separated keys
    elif isinstance(root_value, list):
        yield root_key, ','.join(
            ":".join(str(nested_value[1]) for nested_value in flattened_values(i, lv))
            for (i, lv) in enumerate(root_value)
        )
    # Rest of the values are left as they are
    else:
        yield root_key, root_value


def all_items_of(resource):
    """Generator for iterating through all of the flattened items contained
    inside the resource w.r.t :pkey:`resources` specification.

    Generator iterates through all of the items contained in the `resource` in
    flattened form (i.e. it does not contain nested dictionaries). Resources
    should be w.r.t :pkey:`resources` specification.

    E.g. the following resource:

    .. code-block:: json

        {
            "type": "memory",
            "amount": 4,
            "uid": {
                "source": "../memory_collect_test.c",
                "function": "main",
                "line": 22
            }
        }

    yields the following stream of resources::

        ("type", "memory")
        ("amount", 4)
        ("uid", "../memory_collect_test.c:main:22")
        ("uid:source", "../memory_collect_test.c")
        ("uid:function", "main")
        ("uid:line": 22)

    :param dict resource: dictionary representing one resource
        w.r.t :pkey:`resources`
    :returns: iterable stream of ``(str, value)`` pairs, where the ``value`` is
        flattened to either a `string`, or `decimal` representation and ``str``
        corresponds to the key of the item
    """
    for key, value in resource.items():
        for flattened_key, flattened_value in flattened_values(key, value):
            yield flattened_key, flattened_value


def all_resource_fields_of(profile):
    """Generator for iterating through all of the fields (both flattened and
    original) that are occuring in the resources.

    Generator iterates through all of the resources and checks their flattened
    keys. In case some of the keys were not yet processed, they are yielded.

    E.g. considering the example profiles from :pkey:`resources`, the function
    yields the following for `memory`, `time` and `complexity` profiles
    respectively (considering we convert the stream to list)::

        memory_resource_fields = [
            'type', 'address', 'amount', 'uid:function', 'uid:source',
            'uid:line', 'uid', 'trace', 'subtype'
        ]
        time_resource_fields = [
            'type', 'amount', 'uid'
        ]
        complexity_resource_fields = [
            'type', 'amount', 'structure-unit-size', 'subtype', 'uid'
        ]

    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :returns: iterable stream of resource field keys represented as `str`
    """
    resource_fields = set()
    for (_, resource) in all_resources_of(profile):
        for key, __ in all_items_of(resource):
            if key not in resource_fields:
                resource_fields.add(key)
                yield key


def all_numerical_resource_fields_of(profile):
    """Generator for iterating through all of the fields (both flattened and
    original) that are occuring in the resources and takes as domain integer
    values.

    Generator iterates through all of the resources and checks their flattened
    keys and yields them in case they were not yet processed. If the instance
    of the key does not contain integer values, it is skipped.

    E.g. considering the example profiles from :pkey:`resources`, the function
    yields the following for `memory`, `time` and `complexity` profiles
    respectively (considering we convert the stream to list)::

        memory_num_resource_fields = ['address', 'amount', 'uid:line']
        time_num_resource_fields = ['amount']
        complexity_num_resource_fields = ['amount', 'structure-unit-size']

    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :returns: iterable stream of resource fields key as `str`, that takes
        integer values
    """
    resource_fields = set()
    exclude_fields = set()
    for (_, resource) in all_resources_of(profile):
        for key, value in all_items_of(resource):
            # Instances that are not numbers are removed from the resource fields (i.e. there was
            # some inconsistency between value) and added to exclude for future usages
            if not isinstance(value, numbers.Number):
                resource_fields.discard(value)
                exclude_fields.add(value)
            # If we previously encountered incorrect non-numeric value for the key, we do not add
            # it as a numeric key
            elif value not in exclude_fields:
                resource_fields.add(key)

    # Yield the stream of the keys
    for key in resource_fields:
        yield key


def unique_resource_values_of(profile, resource_key):
    """Generator of all unique key values occurring in the resources, w.r.t.
    :pkey:`resources` specification of resources.

    Iterates through all of the values of given ``resource_keys`` and yields
    only unique values. Note that the key can contain ':' symbol indicating
    another level of dictionary hierarchy or '::' for specifying keys in list
    or set level, e.g. in case of `traces` one uses ``trace::function``.

    E.g. considering the example profiles from :pkey:`resources`, the function
    yields the following for `memory`, `time` and `complexity` profiles stored
    in variables ``mprof``, ``tprof`` and ``cprof`` respectively::

        >>> list(query.unique_resource_values_of(mprof, 'subtype')
        ['malloc', 'free']
        >>> list(query.unique_resource_values_of(tprof, 'amount')
        [0.616, 0.500, 0.125]
        >>> list(query.unique_resource_values_of(cprof, 'uid')
        ['SLList_init(SLList*)', 'SLList_search(SLList*, int)',
         'SLList_insert(SLList*, int)', 'SLList_destroy(SLList*)']

    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :param str resource_key: the resources key identifier whose unique values
        will be iterated
    :returns: iterable stream of unique resource key values
    """
    for value in _unique_values_generator(profile, resource_key, all_resources_of):
        yield value


def all_key_values_of(resource, resource_key):
    """Generator of all (not essentially unique) key values in resource, w.r.t
    :pkey:`resources` specification of resources.

    Iterates through all of the values of given ``resource_key`` and yields
    every value it finds. Note that the key can contain ':' symbol indicating
    another level of dictionary hierarchy or '::' for specifying keys in list
    or set level, e.g. in case of `traces` one uses ``trace::function``.

    E.g. considering the example profiles from :pkey:`resources` and the
    resources ``mres`` from the profile of `memory` type, we can obtain all of
    the values of ``trace::function`` key as follows::

        >>> query.all_key_values_of(mres, 'trace::function')
        ['free', 'main', '__libc_start_main', '_start']

    Note that this is mostly useful for iterating through list or nested
    dictionaries.

    :param dict resource: dictionary representing one resource
        w.r.t :pkey:`resources`
    :param str resource_key: the resources key identifier whose unique values
        will be iterated
    :returns: iterable stream of all resource key values
    """
    # Convert the key identifier to iterable hierarchy
    key_hierarchy = resource_key.split(":")

    # Iterate the hierarchy
    for level_idx, key_level in enumerate(key_hierarchy):
        if key_level == '' and isinstance(resource, (list, set)):
            # The level is list, iterate all the members recursively
            for item in resource:
                for result in all_key_values_of(item, ':'.join(key_hierarchy[level_idx + 1:])):
                    yield result
            return
        elif key_level in resource:
            # The level is dict, find key
            resource = resource[key_level]
        else:
            # No match
            return
    yield resource


def all_models_of(profile):
    """Generator of all 'models' records from the performance profile w.r.t.
    :ref:`profile-spec`.

    Takes a profile, postprocessed by :ref:`postprocessors-regression-analysis`
    and iterates through all of its models (for more details about models refer
    to :pkey:`models` or :ref:`postprocessors-regression-analysis`).

    E.g. given some complexity profile ``complexity_prof``, we can iterate its
    models as follows:

        >>> gen = query.all_models_of(complexity_prof)
        >>> gen.__next__()
        (0, {'x_interval_start': 0, 'model': 'constant', 'method': 'full',
        'coeffs': [{'name': 'b0', 'value': 0.5644496762801648}, {'name': 'b1',
        'value': 0.0}], 'uid': 'SLList_insert(SLList*, int)', 'r_square': 0.0,
        'x_interval_end': 11892})
        >>> gen.__next__()
        (1, {'x_interval_start': 0, 'model': 'exponential', 'method': 'full',
        'coeffs': [{'name': 'b0', 'value': 0.9909792049684152}, {'name': 'b1',
        'value': 1.000004056250301}], 'uid': 'SLList_insert(SLList*, int)',
        'r_square': 0.007076437903106431, 'x_interval_end': 11892})


    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :returns: iterable stream of ``(int, dict)`` pairs, where first yields the
        positional number of model and latter correponds to one 'models'
        record (for more details about models refer to :pkey:`models` or
        :ref:`postprocessors-regression-analysis`)
    """
    # Get models if any
    try:
        models = profile.get('global', {}).get('models', [])
    except AttributeError:
        # global is not dict-like type with get method
        raise exceptions.IncorrectProfileFormatException(
            'profile', "'global' is not a dictionary") from None

    for model_idx, model in enumerate(models):
        yield model_idx, model


def unique_model_values_of(profile, model_key):
    """Generator of all unique key values occurring in the models in the
    resources of given performance profile w.r.t. :ref:`profile-spec`.

    Iterates through all of the values of given ``resource_keys`` and yields
    only unique values. Note that the key can contain ':' symbol indicating
    another level of dictionary hierarchy or '::' for specifying keys in list
    or set level, e.g. in case of `traces` one uses ``trace::function``.  For
    more details about the specification of models refer to :pkey:`models` or
    :ref:`postprocessors-regression-analysis`).

    E.g. given some complexity profile ``complexity_prof``, we can obtain
    unique values of keys from `models` as follows:

        >>> list(query.unique_model_values_of(complexity_prof, 'model')
        ['constant', 'exponential', 'linear', 'logarithmic', 'quadratic']
        >>> list(query.unique_model_values_of(cprof, 'r_square'))
        [0.0, 0.007076437903106431, 0.0017560012128507133,
         0.0008704119815403224, 0.003480627284909902, 0.001977866710139782,
         0.8391363620083871, 0.9840099999298596, 0.7283427343995424,
         0.9709120064750161, 0.9305786182556899]

    :param dict profile: performance profile w.r.t :ref:`profile-spec`
    :param str model_key: key identifier from `models` for which we query
        its unique values
    :returns: iterable stream of unique model key values
    """
    for value in _unique_values_generator(profile, model_key, all_models_of):
        yield value


def _unique_values_generator(profile, key, blocks_gen):
    """Generator of all unique values of 'key' occurring in the profile blocks generated by
    'blocks_gen'.

    :param dict profile: valid profile with models
    :param str key: the key identifier whose unique values are returned
    :param iterable blocks_gen: the data blocks generator (e.g. all_resources_of)
    :returns iterable: stream of unique key values
    """
    # value can be dict, list, set etc and not only simple type, thus the list
    unique_values = list()
    for (_, resource) in blocks_gen(profile):
        # Get all values the key contains
        for value in all_key_values_of(resource, key):
            # Return only the unique ones
            if value not in unique_values:
                unique_values.append(value)
                yield value

# Todo: add optimized version for multiple key search in one go? Need to discuss interface etc.
