from abc import ABC, abstractmethod
from typing import Any

from vidur.types import BaseIntEnum


# 抽象基类，用于管理和注册不同类型的对象
class BaseRegistry(ABC):
    # 表示注册表中的键类型
    _key_class = BaseIntEnum

    # 初始化子类：特殊方法，当一个类派生自 BaseRegistry 时会自动调用
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 用于存储注册的键值对
        cls._registry = {}

    # 将一个实现类注册到注册表中
    # 键是枚举类型，值是一个类
    @classmethod
    def register(cls, key: BaseIntEnum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class

    # 从注册表中移除一个实现类
    @classmethod
    def unregister(cls, key: BaseIntEnum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]

    # 从注册表中获取一个类，以 *args 和 **kwargs 作为构造函数参数。
    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        # 得到类后用参数进行调用，进行构造
        # 获取的是构造函数
        return cls._registry[key](*args, **kwargs)

    # 从注册表中获取一个实现类的类
    @classmethod
    def get_class(cls, key: BaseIntEnum) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key]

    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> BaseIntEnum:
        pass

    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)
