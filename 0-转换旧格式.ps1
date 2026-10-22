# -*- coding: utf-8 -*-
# 0-转换旧格式.ps1
# 用本机 Office/WPS 的 COM 组件把知识库中的旧版二进制文档转换为现代格式：
#   .doc -> .docx （Word COM，wdFormatXMLDocument=12）
#   .ppt -> .pptx （PowerPoint COM，ppSaveAsOpenXMLPresentation=24）
# 转换产物输出到 converted/，并生成 manifest.json 记录 原路径->转换后文件 的映射。
# 已存在的转换结果会跳过（可删除 converted/ 强制重转）。

$ErrorActionPreference = 'Stop'
# 路径全部相对本脚本解析（$PSScriptRoot = 8-金标扩充版本/），换机/换用户不再失效
$kbRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "迪士尼RAG知识库（完整）"
$outDir = Join-Path $PSScriptRoot "converted"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$manifest = @()

# ---------- .doc -> .docx ----------
$docFiles = Get-ChildItem -Path $kbRoot -Recurse -File -Filter *.doc
Write-Host "发现 $($docFiles.Count) 个 .doc 文件，启动 Word COM ..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try { $word.DisplayAlerts = 0 } catch {}
try {
    foreach ($f in $docFiles) {
        $target = Join-Path $outDir ($f.BaseName + '.docx')
        if (Test-Path $target) {
            Write-Host "  跳过(已存在): $($f.Name)"
        } else {
            Write-Host "  转换: $($f.Name)"
            $doc = $word.Documents.Open($f.FullName, $false, $true)  # ConfirmConversions=false, ReadOnly=true
            $doc.SaveAs2($target, 12)
            $doc.Close($false)
        }
        $manifest += [pscustomobject]@{ original = $f.FullName; converted = $target; kind = 'doc->docx' }
    }
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

# ---------- .ppt -> .pptx ----------
$pptFiles = Get-ChildItem -Path $kbRoot -Recurse -File -Filter *.ppt
Write-Host "发现 $($pptFiles.Count) 个 .ppt 文件，启动 PowerPoint COM ..."
$ppt = New-Object -ComObject PowerPoint.Application
try {
    foreach ($f in $pptFiles) {
        $target = Join-Path $outDir ($f.BaseName + '.pptx')
        if (Test-Path $target) {
            Write-Host "  跳过(已存在): $($f.Name)"
        } else {
            Write-Host "  转换: $($f.Name)"
            # Open(FileName, ReadOnly=msoTrue(-1), Untitled=msoFalse(0), WithWindow=msoFalse(0))
            $pres = $ppt.Presentations.Open($f.FullName, -1, 0, 0)
            $pres.SaveAs($target, 24)
            $pres.Close()
        }
        $manifest += [pscustomobject]@{ original = $f.FullName; converted = $target; kind = 'ppt->pptx' }
    }
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}

$manifest | ConvertTo-Json | Set-Content -Path (Join-Path $outDir 'manifest.json') -Encoding UTF8
Write-Host "完成，共 $($manifest.Count) 个映射，清单已保存: converted\manifest.json"
