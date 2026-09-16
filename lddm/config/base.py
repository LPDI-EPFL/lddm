import logging
from dataclasses import dataclass, fields, asdict
import typing


def nested_any_subclass(field_type, class_or_tuple):
    unsubscripted_type = typing.get_origin(field_type) or field_type
    if isinstance(unsubscripted_type, type) and \
        issubclass(unsubscripted_type, class_or_tuple):
        return True
    for x in typing.get_args(field_type):
        if nested_any_subclass(x, class_or_tuple):
            return True
    return False


def nested_any_instance(obj, class_or_tuple):
    if isinstance(obj, class_or_tuple):
        return True
    if not isinstance(obj, (list, tuple, dict)):
        return False
    _elements = obj.values() if isinstance(obj, dict) else obj
    for x in _elements:
        if nested_any_instance(x, class_or_tuple):
            return True
    return False 


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, config_as_dict, strict=True):
        """Build nested config recursively."""

        _fields = {f.name: f for f in fields(cls)}

        if not strict:
            for k in list(config_as_dict):
                if not k in _fields:
                    print(f"[WARNING, {cls.__name__}] Ignoring config parameter {k}.")
                    del config_as_dict[k]

        
        for k, v in config_as_dict.items():
            if k not in _fields:
                continue

            field_type = _fields[k].type

            # Skip empty field
            if v is None:
                pass

            # Standard recursion
            elif isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                config_as_dict[k] = _fields[k].type.from_dict(v, strict=strict)

            # Iterable/mapping types
            elif nested_any_subclass(field_type, BaseConfig) and not nested_any_instance(v, BaseConfig):
                # Iterables and mapping objects can be nested in many 
                # different ways. Here, I assume the attribute has already 
                # been handled correctly in a child class if at least one 
                # BaseConfig-derived dataclass is found in the attribute.
                raise NotImplementedError(
                    f"{cls.__name__}.{k} cannot be parsed automatically. "
                    f"Please update `{cls.__name__}.from_dict` and set its field "
                    f"`{k}` before calling super().from_dict."
                )
            else:
                pass  # config_as_dict[k] = v
        
        return cls(**config_as_dict)
    
    def to_dict(self):
        return asdict(self)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key) -> bool:
        return hasattr(self, key)
    
    def update(self, other: dict) -> None:
        for key, value in other.items():
            if isinstance(self[key], BaseConfig):
                self[key].update(value)
            else:
                logging.debug(f"Replace parameter {type(self).__name__}.{key}: {self[key]} -> {value}")
                self[key] = value

    def get_diff(self, other: "BaseConfig") -> dict[str, tuple]:
        """
        Compare two dataclass instances of the same type.
        Returns a dict of {field_name: (value_in_self, value_in_other)} for all differing fields.
        """
        if type(self) is not type(other):
            raise TypeError(f"Cannot compare different types: {type(self)} vs {type(other)}")

        return {
            f.name: (getattr(self, f.name, None), getattr(other, f.name, None))
            for f in fields(self)
            if getattr(self, f.name, None) != getattr(other, f.name, None)
        }
