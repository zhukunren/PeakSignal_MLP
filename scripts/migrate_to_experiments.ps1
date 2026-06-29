# ============================================================================
# 实验文件迁移脚本
# 用途：将实验性脚本、数据和模型迁移到 experiments/ 目录
# 日期：2026-06-29
# ============================================================================

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取项目根目录
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  实验文件迁移脚本" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 阶段1：创建目录结构
# ============================================================================
Write-Host "[阶段1] 创建实验目录结构..." -ForegroundColor Yellow

$directories = @(
    "experiments",
    "experiments/scripts",
    "experiments/scripts/training",
    "experiments/scripts/evaluation",
    "experiments/scripts/visualization",
    "experiments/scripts/models",
    "experiments/data",
    "experiments/data/raw",
    "experiments/data/cache",
    "experiments/data/temp",
    "experiments/models",
    "experiments/models/base",
    "experiments/models/event_regime",
    "experiments/models/optimized",
    "experiments/results",
    "experiments/results/reports",
    "experiments/results/charts",
    "experiments/results/logs"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $ROOT $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
        Write-Host "  ✓ 创建: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ⊙ 已存在: $dir" -ForegroundColor Gray
    }
}

Write-Host ""

# ============================================================================
# 阶段2：迁移脚本文件
# ============================================================================
Write-Host "[阶段2] 迁移脚本文件..." -ForegroundColor Yellow

# 定义迁移映射：源路径 -> 目标路径
$scriptMigrations = @{
    # 训练脚本
    "scripts/batchtraining.py" = "experiments/scripts/training/"
    "scripts/run_fixed_feature_training_check.py" = "experiments/scripts/training/"
    "scripts/run_training_combo_check.py" = "experiments/scripts/training/"
    "scripts/run_cached_combo_training_check.py" = "experiments/scripts/training/"
    "scripts/train_combo_round_worker.py" = "experiments/scripts/training/"
    "scripts/save_base_round008_model.py" = "experiments/scripts/training/"

    # 评估脚本
    "scripts/strict_oos_event_validation.py" = "experiments/scripts/evaluation/"
    "scripts/optimize_saved_model_thresholds.py" = "experiments/scripts/evaluation/"

    # 可视化脚本
    "scripts/generate_best_combo_chart.py" = "experiments/scripts/visualization/"
    "scripts/generate_base_best_cached_chart.py" = "experiments/scripts/visualization/"

    # 模型训练脚本
    "scripts/train_event_regime_model.py" = "experiments/scripts/models/"
}

foreach ($migration in $scriptMigrations.GetEnumerator()) {
    $source = Join-Path $ROOT $migration.Key
    $destDir = Join-Path $ROOT $migration.Value

    if (Test-Path $source) {
        $fileName = Split-Path $source -Leaf
        $dest = Join-Path $destDir $fileName

        if (Test-Path $dest) {
            Write-Host "  ⊙ 跳过（目标已存在）: $($migration.Key)" -ForegroundColor Gray
        } else {
            Move-Item -Path $source -Destination $dest -Force
            Write-Host "  ✓ 迁移: $($migration.Key) -> $($migration.Value)" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⚠ 源文件不存在: $($migration.Key)" -ForegroundColor DarkYellow
    }
}

Write-Host ""

# ============================================================================
# 阶段3：迁移数据和模型
# ============================================================================
Write-Host "[阶段3] 迁移数据和模型..." -ForegroundColor Yellow

# 数据文件迁移
$dataFile = Join-Path $ROOT "完整数据.csv"
$dataTarget = Join-Path $ROOT "experiments/data/raw/完整数据.csv"
if (Test-Path $dataFile) {
    if (Test-Path $dataTarget) {
        Write-Host "  ⊙ 跳过: 完整数据.csv（目标已存在）" -ForegroundColor Gray
    } else {
        Move-Item -Path $dataFile -Destination $dataTarget -Force
        Write-Host "  ✓ 迁移: 完整数据.csv -> experiments/data/raw/" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ 未找到: 完整数据.csv" -ForegroundColor DarkYellow
}

# 缓存目录迁移
$cacheDir = Join-Path $ROOT "fixed_feature_combo_cache"
$cacheTarget = Join-Path $ROOT "experiments/data/cache/fixed_feature_combo_cache"
if (Test-Path $cacheDir) {
    if (Test-Path $cacheTarget) {
        Write-Host "  ⊙ 跳过: fixed_feature_combo_cache/（目标已存在）" -ForegroundColor Gray
    } else {
        Move-Item -Path $cacheDir -Destination $cacheTarget -Force
        Write-Host "  ✓ 迁移: fixed_feature_combo_cache/ -> experiments/data/cache/" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ 未找到: fixed_feature_combo_cache/" -ForegroundColor DarkYellow
}

# 模型文件迁移
$modelMigrations = @{
    "base_98pct_round008_model.pkl" = "experiments/models/base/"
    "saved_models/event_regime_hgbr_2021_present_model.pkl" = "experiments/models/event_regime/"
    "saved_models/strict_oos_event_regime_model.pkl" = "experiments/models/event_regime/"
    "saved_models/optimized_2021_present_threshold_model.pkl" = "experiments/models/optimized/"
}

foreach ($migration in $modelMigrations.GetEnumerator()) {
    $source = Join-Path $ROOT $migration.Key
    $destDir = Join-Path $ROOT $migration.Value

    if (Test-Path $source) {
        $fileName = Split-Path $source -Leaf
        $dest = Join-Path $destDir $fileName

        if (Test-Path $dest) {
            Write-Host "  ⊙ 跳过: $($migration.Key)（目标已存在）" -ForegroundColor Gray
        } else {
            Move-Item -Path $source -Destination $dest -Force
            Write-Host "  ✓ 迁移: $($migration.Key) -> $($migration.Value)" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⚠ 未找到: $($migration.Key)" -ForegroundColor DarkYellow
    }
}

Write-Host ""

# ============================================================================
# 阶段4：迁移结果文件
# ============================================================================
Write-Host "[阶段4] 迁移结果文件..." -ForegroundColor Yellow

# JSON 报告迁移
$reportFiles = Get-ChildItem -Path (Join-Path $ROOT "saved_models") -Filter "*_report.json" -ErrorAction SilentlyContinue
foreach ($file in $reportFiles) {
    $dest = Join-Path $ROOT "experiments/results/reports/$($file.Name)"
    if (Test-Path $dest) {
        Write-Host "  ⊙ 跳过: saved_models/$($file.Name)（目标已存在）" -ForegroundColor Gray
    } else {
        Move-Item -Path $file.FullName -Destination $dest -Force
        Write-Host "  ✓ 迁移: saved_models/$($file.Name) -> experiments/results/reports/" -ForegroundColor Green
    }
}

# HTML 图表迁移
$htmlFiles = Get-ChildItem -Path $ROOT -Filter "*.html" -ErrorAction SilentlyContinue
foreach ($file in $htmlFiles) {
    $dest = Join-Path $ROOT "experiments/results/charts/$($file.Name)"
    if (Test-Path $dest) {
        Write-Host "  ⊙ 跳过: $($file.Name)（目标已存在）" -ForegroundColor Gray
    } else {
        Move-Item -Path $file.FullName -Destination $dest -Force
        Write-Host "  ✓ 迁移: $($file.Name) -> experiments/results/charts/" -ForegroundColor Green
    }
}

# LOG 文件迁移
$logFiles = Get-ChildItem -Path $ROOT -Filter "*.log" -ErrorAction SilentlyContinue
foreach ($file in $logFiles) {
    $dest = Join-Path $ROOT "experiments/results/logs/$($file.Name)"
    if (Test-Path $dest) {
        Write-Host "  ⊙ 跳过: $($file.Name)（目标已存在）" -ForegroundColor Gray
    } else {
        Move-Item -Path $file.FullName -Destination $dest -Force
        Write-Host "  ✓ 迁移: $($file.Name) -> experiments/results/logs/" -ForegroundColor Green
    }
}

Write-Host ""

# ============================================================================
# 阶段5：更新 .gitignore
# ============================================================================
Write-Host "[阶段5] 更新 .gitignore..." -ForegroundColor Yellow

$gitignorePath = Join-Path $ROOT ".gitignore"
$gitignoreContent = Get-Content $gitignorePath -Raw -ErrorAction SilentlyContinue

if ($gitignoreContent -notlike "*experiments/*") {
    $appendContent = @"

# Experiments directory (entire folder excluded)
experiments/
!experiments/README.md
"@
    Add-Content -Path $gitignorePath -Value $appendContent
    Write-Host "  ✓ 已添加 experiments/ 到 .gitignore" -ForegroundColor Green
} else {
    Write-Host "  ⊙ .gitignore 已包含 experiments/ 规则" -ForegroundColor Gray
}

Write-Host ""

# ============================================================================
# 完成
# ============================================================================
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  迁移完成！" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "后续步骤：" -ForegroundColor Yellow
Write-Host "1. 检查 experiments/ 目录结构是否正确"
Write-Host "2. 更新实验脚本中的路径引用（ROOT_DIR 等）"
Write-Host "3. 测试运行关键脚本确保路径正确"
Write-Host "4. 可选：删除空的 scripts/ 目录（如果所有脚本都已迁移）"
Write-Host ""
Write-Host "运行测试脚本示例：" -ForegroundColor Yellow
Write-Host "  python experiments/scripts/models/train_event_regime_model.py --help"
Write-Host ""
