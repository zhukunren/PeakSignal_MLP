# 配置管理方案

## 🎯 问题分析

**当前配置散落各处**：
```python
# app.py
TARGET_REPRO_SEED_BASE = 7300
TARGET_REPRO_BEST_ROUND = 8
num_rounds = 10

# ml_trader/models/trainer.py
patience = 3
val_size = 0.2

# ml_trader/data/preprocessor.py
threshold = 0.95
max_components = 100
```

**问题**：
- ❌ 硬编码分散
- ❌ 修改配置需要改代码
- ❌ 不同环境难以切换
- ❌ 无法版本控制配置历史

---

## 🎨 解决方案：分层配置

```
config/
├── default.yaml           # 默认配置
├── development.yaml       # 开发环境配置
├── production.yaml        # 生产环境配置
└── local.yaml            # 本地配置（不提交到 Git）
```

---

## 📁 配置文件结构

### default.yaml - 默认配置

```yaml
# 项目基础配置
app:
  name: "东吴秀享AI超额收益系统"
  version: "0.1.0"
  environment: "development"

# 训练配置
training:
  seed_base: 7300
  best_round: 8
  num_rounds: 10
  default_train_start: "2000-01-01"
  default_train_end: "2020-12-31"
  
  # 模型训练参数
  patience: 3
  val_size: 0.2
  test_size: 0.2
  early_stopping_rounds: 10

# 预测配置
prediction:
  default_pred_start: "2021-01-01"
  use_best_combo: true
  combination_search:
    enable_chase: false
    enable_stop_loss: false
    enable_change_signal: false
    n_buy: 1
    n_sell: 1
    n_newhigh: 60

# 特征工程配置
features:
  mixture_depth: 1
  correlation_threshold: 0.95
  variance_threshold: 0.0001
  pca:
    enable: true
    max_components: 100

# 数据配置
data:
  cache_dir: "cache/"
  models_dir: "saved_models/"
  experiments_dir: "experiments/"
  
  # Tushare 配置（从环境变量读取）
  tushare_token: ${TUSHARE_TOKEN}

# 回测配置
backtest:
  initial_capital: 1000000
  commission_rate: 0.0003
  
# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/app.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

---

### production.yaml - 生产环境配置

```yaml
# 继承 default.yaml，只覆盖差异部分
app:
  environment: "production"

logging:
  level: "WARNING"
  file: "/var/log/ml_trader/app.log"

data:
  cache_dir: "/data/ml_trader/cache/"
  models_dir: "/data/ml_trader/models/"
```

---

### local.yaml - 本地配置（示例）

```yaml
# 本地开发者个人配置
# 此文件不提交到 Git

data:
  tushare_token: "your_actual_token_here"

training:
  num_rounds: 2  # 开发时快速测试

logging:
  level: "DEBUG"
```

---

## 🔧 配置加载代码

### config_loader.py

```python
# ml_trader/config_loader.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 1. 加载 default.yaml
        default_config = self._load_yaml("default.yaml")
        
        # 2. 根据环境加载对应配置
        env = os.getenv("APP_ENV", "development")
        env_config = self._load_yaml(f"{env}.yaml")
        
        # 3. 加载本地配置（如果存在）
        local_config = self._load_yaml("local.yaml", required=False)
        
        # 4. 合并配置（后者覆盖前者）
        config = self._merge_dicts(default_config, env_config)
        if local_config:
            config = self._merge_dicts(config, local_config)
        
        # 5. 解析环境变量
        config = self._resolve_env_vars(config)
        
        return config
    
    def _load_yaml(self, filename: str, required: bool = True) -> Dict:
        """加载 YAML 文件"""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            if required:
                raise FileNotFoundError(f"配置文件不存在: {filepath}")
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or 
    
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
        """获取配置值（支持点号路径）
        
        Example:
            config.get("training.seed_base")  # 7300
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# 全局配置实例
config = ConfigLoader()
```

---

## 📝 使用示例

### 在代码中使用配置

```python
# app.py
from ml_trader.config_loader import config

# 读取配置
seed_base = config.get("training.seed_base")
num_rounds = config.get("training.num_rounds")
initial_capital = config.get("backtest.initial_capital")

# 带默认值
debug_mode = config.get("app.debug", False)
```

```python
# ml_trader/models/trainer.py
from ml_trader.config_loader import config

def train_model(...):
    patience = config.get("training.patience", 3)
    val_size = config.get("training.val_size", 0.2)
    ...
```

---

## 🔐 环境变量管理

### .env.example - 环境变量模板

```bash
# Tushare API Token
TUSHARE_TOKEN=your_token_here

# 应用环境 (development/production)
APP_ENV=development

# 日志级别
LOG_LEVEL=INFO
```

### .env - 实际环境变量（不提交到 Git）

```bash
TUSHARE_TOKEN=abc123xyz456...
APP_ENV=development
LOG_LEVEL=DEBUG
```

### 加载环境变量

```python
# app.py 或 ml_trader/__init__.py
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 现在可以使用环境变量
token = os.getenv("TUSHARE_TOKEN")
```

---

## 🎛️ 配置验证

```python
# ml_trader/config_validator.py
from pydantic import BaseModel, Field, validator
from typing import Optional

class TrainingConfig(BaseModel):
    """训练配置验证"""
    seed_base: int = Field(ge=0, description="随机种子基准")
    best_round: int = Field(ge=1, le=100, description="最佳轮次")
    num_rounds: int = Field(ge=1, le=100, description="训练轮数")
    patience: int = Field(ge=1, description="早停耐心值")
    val_size: float = Field(gt=0, lt=1, description="验证集比例")
    
    @validator('best_round')
    def validate_best_round(cls, v, values):
        if 'num_rounds' in values and v > values['num_rounds']:
            raise ValueError("best_round 不能大于 num_rounds")
        return v

class Config(BaseModel):
    """总配置验证"""
    training: TrainingConfig
    # ... 其他配置

def validate_config(config_dict: dict) -> Config:
    """验证配置"""
    try:
        return Config(**config_dict)
    except Exception as e:
        raise ValueError(f"配置验证失败: {e}")
```

---

## 🔄 配置热更新（可选）

```python
# ml_trader/config_loader.py (扩展)
import time
import threading

class ConfigLoader:
    def __init__(self, config_dir: str = "config", auto_reload: bool = False):
        self.config_dir = Path(config_dir)
        self.config = self._load_config()
        self._last_mtime = {}
        
        if auto_reload:
            self._start_watch_thread()
    
    def _start_watch_thread(self):
        """启动配置文件监控线程"""
        def watch():
            while True:
                time.sleep(5)  # 每5秒检查一次
                if self._config_changed():
                    print("[Config] 检测到配置变化，重新加载...")
                    self.config = self._load_config()
        
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
    
    def _config_changed(self) -> bool:
        """检查配置文件是否修改"""
        for yaml_file in self.config_dir.glob("*.yaml"):
            mtime = yaml_file.stat().st_mtime
            if yaml_file not in self._last_mtime or self._last_mtime[yaml_file] != mtime:
                self._last_mtime[yaml_file] = mtime
                return True
        return False
```

---

## 📋 迁移清单

### 需要迁移的配置项

| 当前位置 | 配置项 | 目标位置 |
|---------|--------|----------|
| app.py:28 | TARGET_REPRO_SEED_BASE | config/default.yaml: training.seed_base |
| app.py:29 | TARGET_REPRO_BEST_ROUND | config/default.yaml: training.best_round |
| app.py:895 | num_rounds | config/default.yaml: training.num_rounds |
| trainer.py | patience | config/default.yaml: training.patience |
| preprocessor.py | threshold | config/default.yaml: features.correlation_threshold |

---

## ⏱️ 实施计划

- **第1天**: 创建配置文件结构和加载器
- **第2天**: 迁移 app.py 中的配置
- **第3天**: 迁移 ml_trader/ 中的配置
- **第4天**: 测试和文档

**总计**: 4天

---

## ✅ 验收标准

1. ✅ 所有硬编码配置移至 YAML
2. ✅ 支持开发/生产环境切换
3. ✅ 敏感信息使用环境变量
4. ✅ 配置变更无需修改代码
5. ✅ 提供配置文档和示例

---

**优先级**: 🟡 中  
**复杂度**: 低  
**收益**: 高（便于部署和维护）
