[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,
    [switch]$RebuildDeps,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = 'RX5950XT/Miroxen'
$root = Split-Path -Parent $PSScriptRoot
$tag = "v$Version"
$stage = $null

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed (exit $LASTEXITCODE)." }
}

# 不用 Get-FileHash：從 pwsh 7 或 Git Bash 呼叫 5.1 時 PSModulePath 會讓它載不到
function Get-Sha256 {
    param([string]$Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '') }
    finally { $stream.Dispose(); $sha.Dispose() }
}

function Copy-PhysicalFile {
    param([string]$Path, [string]$Destination)
    $file = Get-Item -LiteralPath $Path
    $seen = @{}
    while ($file.LinkType -eq 'SymbolicLink') {
        if ($seen.ContainsKey($file.FullName)) { throw "Symbolic link cycle: $Path" }
        $seen[$file.FullName] = $true
        $target = @($file.Target)[0]
        if (![IO.Path]::IsPathRooted($target)) { $target = Join-Path $file.DirectoryName $target }
        $file = Get-Item -LiteralPath $target
    }
    if ($file.PSIsContainer -or $file.Length -eq 0) { throw "Invalid file: $Path" }
    Copy-Item -LiteralPath $file.FullName -Destination $Destination
}

function Build-Dependencies {
    param([string]$Destination)
    $cache = Get-Content -LiteralPath (Join-Path $root 'build/CMakeCache.txt')
    $compilerLine = @($cache | Where-Object { $_ -match '^CMAKE_CUDA_COMPILER:[^=]+=' })
    if ($compilerLine.Count -ne 1) { throw 'CMAKE_CUDA_COMPILER not found in build/CMakeCache.txt.' }
    $compiler = ($compilerLine[0] -split '=', 2)[1].Trim('"')
    $cudaBin = Split-Path -Parent $compiler
    if ((Split-Path -Leaf $cudaBin) -eq 'x64') { $cudaBin = Split-Path -Parent $cudaBin }
    foreach ($name in @('cublas64_13.dll', 'cublasLt64_13.dll')) {
        $source = @((Join-Path $cudaBin "x64/$name"), (Join-Path $cudaBin $name)) |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (!$source) { throw "CUDA dependency not found: $name ($cudaBin)" }
        Copy-PhysicalFile $source (Join-Path $Destination $name)
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = Invoke-Checked $vswhere @('-latest', '-products', '*', '-version', '[17.0,18.0)',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
    if (!$vs) { throw 'Visual Studio 2022 C++ tools not found.' }
    $redist = Get-ChildItem -LiteralPath (Join-Path $vs 'VC/Redist/MSVC') -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (!$redist) { throw 'MSVC runtime directory not found.' }
    $crt = Join-Path $redist.FullName 'x64/Microsoft.VC143.CRT'
    foreach ($name in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
        Copy-PhysicalFile (Join-Path $crt $name) (Join-Path $Destination $name)
    }
    foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
        $command = Get-Command $name -CommandType Application | Select-Object -First 1
        Copy-PhysicalFile $command.Source (Join-Path $Destination $name)
    }
    $ffmpegVersion = (Invoke-Checked (Join-Path $Destination 'ffmpeg.exe') @('-version')) -join "`n"
    $ffmpegLicense = (Invoke-Checked (Join-Path $Destination 'ffmpeg.exe') @('-hide_banner', '-L')) -join "`n"
    $notices = @"
第三方元件來源與授權

CUDA cuBLAS / cuBLASLt（NVIDIA）
檔案：cublas64_13.dll、cublasLt64_13.dll
來源：建置快取指定的 $compiler 所屬 CUDA Toolkit。
下載：https://developer.nvidia.com/cuda-downloads
授權：https://docs.nvidia.com/cuda/eula/index.html

Microsoft Visual C++ runtime（Microsoft）
檔案：msvcp140.dll、vcruntime140.dll、vcruntime140_1.dll
來源：$crt
授權與可散布清單：https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution
授權條款：https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/

FFmpeg / FFprobe（FFmpeg contributors，GPL 建置）
來源：PATH 中的 ffmpeg.exe、ffprobe.exe，複製前解析符號連結。
ffmpeg：$((Get-Command ffmpeg.exe -CommandType Application | Select-Object -First 1).Source)
ffprobe：$((Get-Command ffprobe.exe -CommandType Application | Select-Object -First 1).Source)
官方原始碼：https://ffmpeg.org/download.html#get-sources
Windows 建置來源（本機 WinGet Gyan.FFmpeg）：https://www.gyan.dev/ffmpeg/builds/
授權說明：https://ffmpeg.org/legal.html
GPL：https://www.gnu.org/licenses/gpl-3.0.html
實際版本、建置選項與授權以以下工具輸出為準；其他 PATH 版本請依其建置來源取得對應原始碼。

$ffmpegVersion

$ffmpegLicense
"@
    [IO.File]::WriteAllText((Join-Path $Destination 'THIRD_PARTY_NOTICES.txt'), $notices, (New-Object Text.UTF8Encoding $false))
}

Push-Location $root
try {
    $dirty = @(Invoke-Checked 'git' @('status', '--porcelain'))
    if ($dirty.Count) {
        if (!$DryRun) { throw 'Working tree is not clean. Commit changes before releasing.' }
        Write-Warning 'Working tree is not clean; DryRun continues with current files.'
    }
    $tags = @(Invoke-Checked 'git' @('for-each-ref', '--format=%(refname:short)', 'refs/tags'))
    if ($tags -contains $tag) { throw "Tag already exists: $tag" }
    if (!$DryRun) {
        $remoteTag = @(Invoke-Checked 'git' @('ls-remote', '--tags', 'origin', "refs/tags/$tag"))
        if ($remoteTag.Count) { throw "Remote tag already exists: $tag" }
    }
    $output = Join-Path $root "out/release/$tag"
    if (Test-Path -LiteralPath $output) { throw "Output already exists; refusing to overwrite: $output" }
    Invoke-Checked (Join-Path $root 'build.bat') @()
    $exe = Join-Path $root 'build/miroxen.exe'
    if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build output not found: $exe" }

    New-Item -ItemType Directory -Path $output | Out-Null
    $stage = Join-Path $output ('.stage-' + [guid]::NewGuid().ToString('N'))
    $appStage = Join-Path $stage 'app'
    New-Item -ItemType Directory -Path $appStage | Out-Null
    Copy-Item -LiteralPath $exe -Destination (Join-Path $appStage 'miroxen.exe')
    Copy-Item -LiteralPath (Join-Path $root 'web') -Destination (Join-Path $appStage 'web') -Recurse
    [IO.File]::WriteAllText((Join-Path $appStage 'VERSION'), "$tag`n", (New-Object Text.UTF8Encoding $false))
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $appZip = Join-Path $output 'miroxen-win-x64.zip'
    $depsZip = Join-Path $output 'miroxen-deps-win-x64.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($appStage, $appZip)

    $previousRelease = $null
    $reuseDeps = $false
    if (!$DryRun) {
        $releases = ConvertFrom-Json ((Invoke-Checked 'gh' @('release', 'list', '--repo', $repo,
            '--exclude-drafts', '--exclude-pre-releases', '--limit', '1', '--json', 'tagName')) -join "`n")
        if ($releases) {
            $previousRelease = $releases[0].tagName
            if (!$RebuildDeps) {
                $release = ConvertFrom-Json ((Invoke-Checked 'gh' @('release', 'view', $previousRelease,
                    '--repo', $repo, '--json', 'assets')) -join "`n")
                $reuseDeps = @($release.assets | Where-Object { $_.name -eq 'miroxen-deps-win-x64.zip' }).Count -gt 0
            }
        }
    }
    if ($reuseDeps) {
        Invoke-Checked 'gh' @('release', 'download', $previousRelease, '--repo', $repo,
            '--pattern', 'miroxen-deps-win-x64.zip', '--dir', $output)
        Write-Host "Reused dependencies from $previousRelease (unchanged ZIP)."
    } else {
        $depsStage = Join-Path $stage 'deps'
        New-Item -ItemType Directory -Path $depsStage | Out-Null
        Build-Dependencies $depsStage
        [IO.Compression.ZipFile]::CreateFromDirectory($depsStage, $depsZip)
    }
    foreach ($zip in @($appZip, $depsZip)) {
        $file = Get-Item -LiteralPath $zip
        if (!$file.Length) { throw "Empty ZIP: $zip" }
        Write-Host ("{0}: {1} bytes`nSHA-256: {2}" -f $file.Name, $file.Length, (Get-Sha256 $zip))
    }
    if ($DryRun) {
        Write-Host "DryRun complete: $output (no Git/GitHub changes or GitHub requests)."
    } else {
        $mergedTags = @(Invoke-Checked 'git' @('for-each-ref', '--merged=HEAD', '--format=%(refname:short)', 'refs/tags'))
        $previousTag = if ($mergedTags.Count) { Invoke-Checked 'git' @('describe', '--tags', '--abbrev=0', 'HEAD') } else { $null }
        $range = if ($previousTag) { "$previousTag..HEAD" } else { 'HEAD' }
        $notes = (Invoke-Checked 'git' @('log', '--oneline', $range)) -join "`n"
        if (!$notes) { $notes = $tag }
        $notesFile = Join-Path $stage 'notes.md'
        [IO.File]::WriteAllText($notesFile, $notes, (New-Object Text.UTF8Encoding $false))
        Invoke-Checked 'git' @('tag', $tag)
        Invoke-Checked 'git' @('push', 'origin', $tag)
        Invoke-Checked 'gh' @('release', 'create', $tag, $appZip, $depsZip, '--repo', $repo, '--title', $tag, '--notes-file', $notesFile)
    }
} catch {
    Write-Error -ErrorRecord $_ -ErrorAction Continue
    exit 1
} finally {
    if ($stage -and (Test-Path -LiteralPath $stage)) {
        $stagePath = [IO.Path]::GetFullPath($stage)
        $outputPath = [IO.Path]::GetFullPath($output) + [IO.Path]::DirectorySeparatorChar
        if (!$stagePath.StartsWith($outputPath, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe staging path.' }
        Remove-Item -LiteralPath $stagePath -Recurse -Force
    }
    Pop-Location
}
