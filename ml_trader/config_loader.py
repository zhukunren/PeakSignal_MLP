"""
配置加载器

从 YAML 文件加载配置，支持环境变量和多环境配置
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 1. 加载 default.yaml
        default_config = self._load_yaml("default.yaml")

        # 2. 根据环境加载对应配置（可选）
        env = os.getenv("APP_ENV", "development")
        env_config_file = f"{env}.yaml"
        if (self.config_dir / env_config_file).exists():
            env_config = self._load_yaml(env_config_file)
            default_config = self._merge_dicts(default_config, env_config)

        # 3. 加载本地配置（如果存在，不提交到 Git）
        if (self.config_dir / "local.yaml").exists():
            local_config = self._load_yaml("local.yaml")
            default_config = self._merge_dicts(default_config, local_config)

        # 4. 解析环境变量
        config = self._resolve_env_vars(default_config)

        return config

    def _load_yaml(self, filename: str) -> Dict:
        """加载 YAML 文件"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """递归合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _resolve_env_vars(self, config: Any) -> Any:
        """解析环境变量 ${VAR_NAME}"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）

        Example:
            config.get("training.seed_base")  # 7300
            config.get("features.mixture_depth", 1)
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取整个配置节"""
        return self.config.get(section, {})


# 全局配置实例
config = ConfigLoader()


# 便捷访问函数
def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config.get(key_path, default)
