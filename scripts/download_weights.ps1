Param()

Write-Host "This script downloads model weight files into the repository. Edit the `\$DOWNLOAD_URLS` array to add real URLs."

$DOWNLOAD_URLS = @(
    'https://example.com/path/to/resnet50.pth'
    'https://example.com/path/to/sam_vit_b.pth'
)

New-Item -ItemType Directory -Path checkpoints -ErrorAction SilentlyContinue | Out-Null
foreach ($url in $DOWNLOAD_URLS) {
    $fname = [System.IO.Path]::GetFileName($url)
    Write-Host "Downloading $fname..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $fname -UseBasicParsing
    } catch {
        Write-Error "Failed to download $url - $_"
        exit 1
    }
    Move-Item -Force $fname checkpoints\
}

Write-Host "Downloaded files moved to checkpoints\". Verify and move to models\" if required."