"""Packaged schemas and operation registry; validation has no weighted runtime imports."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource


def _offline(uri):
    raise NoSuchResource(ref=uri)


@lru_cache(maxsize=1)
def contracts():
    root = files(__package__)
    schemas = {}
    for name in ("step", "workflow", "adapter", "values"):
        schema = json.loads(root.joinpath("schemas", f"{name}-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        schemas[name] = schema
    registry = Registry(retrieve=_offline).with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    operations = json.loads(root.joinpath("operations.json").read_text())
    return schemas, registry, operations


def schema_errors(schema, value):
    _, registry, _ = contracts()
    return sorted(
        Draft202012Validator(schema, registry=registry).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )


def value_schema(type_name):
    schemas, _, _ = contracts()
    return schemas["values"]["$defs"][type_name]
