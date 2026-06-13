Param()

$required = @(
    'checkpoints/sam_vit_b.pth'
    'models/regressor.pkl'
    'models/pca.pkl'
    'models/hybrid_pca_regressor.pkl'
    'models/pca_with_metadata.pkl'
    'models/lgbm_with_metadata.pkl'
    'models/metadata_processor.pkl'
    'models/efficientnet_b3/model.safetensors'
)

$missing = $false
foreach ($path in $required) {
    if (-Not (Test-Path $path)) {
        Write-Host "MISSING: $path" -ForegroundColor Red
        $missing = $true
    } else {
        Write-Host "OK: $path" -ForegroundColor Green
    }
}

if ($missing) {
    Write-Host "`nSome required model weight files are missing. Place them in the correct paths and rerun this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "All required model weight files are present." -ForegroundColor Green
