#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Verify NEODrone Dataset Processing Pipeline Output

.DESCRIPTION
    This script verifies the output of the NEODrone dataset processing pipeline
    to ensure it meets quality standards and matches expected specifications.

.PARAMETER OutputDir
    Output directory containing processed data

.PARAMETER ExpectedFrames
    Expected number of frames

.PARAMETER MetadataFile
    Path to metadata Excel file

.PARAMETER AnnotationsDir
    Path to annotations directory

.EXAMPLE
    .\verify_output.ps1 -OutputDir "processed_data" -ExpectedFrames 50000
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [int]$ExpectedFrames = 0,

    [string]$MetadataFile = "",

    [string]$AnnotationsDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "INFO" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "SUCCESS" { "Cyan" }
        default { "White" }
    }
    
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-DirectoryStructure {
    Write-Log "Testing directory structure..."
    
    $requiredDirs = @(
        "frames",
        "synced",
        "cleaned",
        "annotations",
        "validation",
        "qa_audit"
    )
    
    $missingDirs = @()
    foreach ($dir in $requiredDirs) {
        $dirPath = Join-Path $OutputDir $dir
        if (Test-Path $dirPath) {
            Write-Log "  ✓ Found: $dir"
        } else {
            Write-Log "  ✗ Missing: $dir" -Level "WARNING"
            $missingDirs += $dir
        }
    }
    
    return $missingDirs.Count -eq 0
}

function Test-FrameCount {
    Write-Log "Testing frame count..."
    
    $framesDir = Join-Path $OutputDir "frames"
    if (-not (Test-Path $framesDir)) {
        Write-Log "  Frames directory not found" -Level "ERROR"
        return $false
    }
    
    $frameFiles = Get-ChildItem -Path $framesDir -Recurse -Include *.jpg, *.png
    $actualFrames = $frameFiles.Count
    
    Write-Log "  Actual frames: $actualFrames"
    
    if ($ExpectedFrames -gt 0) {
        Write-Log "  Expected frames: $ExpectedFrames"
        
        $difference = [math]::Abs($actualFrames - $ExpectedFrames)
        $tolerance = [math]::Floor($ExpectedFrames * 0.05)
        
        if ($difference -le $tolerance) {
            Write-Log "  ✓ Frame count within tolerance (±$tolerance)" -Level "SUCCESS"
            return $true
        } else {
            Write-Log "  ✗ Frame count exceeds tolerance" -Level "ERROR"
            return $false
        }
    } else {
        Write-Log "  ✓ Frame count verified (no expected count provided)" -Level "SUCCESS"
        return $true
    }
}

function Test-MetadataCompleteness {
    Write-Log "Testing metadata completeness..."
    
    if ([string]::IsNullOrEmpty($MetadataFile)) {
        $MetadataFile = Join-Path $OutputDir "metadata.xlsx"
    }
    
    if (-not (Test-Path $MetadataFile)) {
        Write-Log "  Metadata file not found: $MetadataFile" -Level "ERROR"
        return $false
    }
    
    try {
        $data = Import-Excel -Path $MetadataFile -ErrorAction Stop
        $totalRecords = $data.Count
        
        if ($totalRecords -eq 0) {
            Write-Log "  ✗ No records found in metadata" -Level "ERROR"
            return $false
        }
        
        $requiredFields = @(
            "flight_id", "timestamp", "latitude", "longitude", "altitude",
            "heading", "pitch", "roll", "gimbal_pitch", "gimbal_roll",
            "camera_zoom", "camera_focus", "acquisition_weather", "acquisition_scene",
            "sensor_type"
        )
        
        $missingFields = @()
        foreach ($field in $requiredFields) {
            if ($field -notin $data.PSObject.Properties.Name) {
                $missingFields += $field
            }
        }
        
        if ($missingFields.Count -gt 0) {
            Write-Log "  ✗ Missing fields: $($missingFields -join ', ')" -Level "ERROR"
            return $false
        }
        
        $completeRecords = 0
        foreach ($record in $data) {
            $isComplete = $true
            foreach ($field in $requiredFields) {
                if ([string]::IsNullOrEmpty($record.$field)) {
                    $isComplete = $false
                    break
                }
            }
            if ($isComplete) {
                $completeRecords++
            }
        }
        
        $completenessRate = ($completeRecords / $totalRecords) * 100
        Write-Log "  Complete records: $completeRecords/$totalRecords ($completenessRate:F2%)"
        
        if ($completenessRate -ge 95) {
            Write-Log "  ✓ Metadata completeness acceptable (≥95%)" -Level "SUCCESS"
            return $true
        } else {
            Write-Log "  ✗ Metadata completeness below threshold" -Level "ERROR"
            return $false
        }
        
    } catch {
        Write-Log "  ✗ Error reading metadata: $_" -Level "ERROR"
        return $false
    }
}

function Test-AnnotationConsistency {
    Write-Log "Testing annotation consistency..."
    
    if ([string]::IsNullOrEmpty($AnnotationsDir)) {
        $AnnotationsDir = Join-Path $OutputDir "annotations"
    }
    
    if (-not (Test-Path $AnnotationsDir)) {
        Write-Log "  Annotations directory not found" -Level "WARNING"
        return $true
    }
    
    $annotationFiles = Get-ChildItem -Path $AnnotationsDir -Filter *.txt
    $totalAnnotations = $annotationFiles.Count
    
    if ($totalAnnotations -eq 0) {
        Write-Log "  No annotation files found" -Level "WARNING"
        return $true
    }
    
    Write-Log "  Total annotation files: $totalAnnotations"
    
    $validAnnotations = 0
    $invalidAnnotations = 0
    
    foreach ($file in $annotationFiles) {
        try {
            $content = Get-Content $file.FullName -Raw
            if ([string]::IsNullOrWhiteSpace($content)) {
                $invalidAnnotations++
            } else {
                $lines = $content -split "`r`n|`n|`r"
                $validLines = 0
                foreach ($line in $lines) {
                    if (-not [string]::IsNullOrWhiteSpace($line)) {
                        $parts = $line -split '\s+'
                        if ($parts.Count -ge 5) {
                            $validLines++
                        }
                    }
                }
                
                if ($validLines -gt 0) {
                    $validAnnotations++
                } else {
                    $invalidAnnotations++
                }
            }
        } catch {
            $invalidAnnotations++
        }
    }
    
    Write-Log "  Valid annotations: $validAnnotations"
    Write-Log "  Invalid annotations: $invalidAnnotations"
    
    $consistencyRate = ($validAnnotations / $totalAnnotations) * 100
    
    if ($consistencyRate -ge 90) {
        Write-Log "  ✓ Annotation consistency acceptable (≥90%)" -Level "SUCCESS"
        return $true
    } else {
        Write-Log "  ✗ Annotation consistency below threshold" -Level "ERROR"
        return $false
    }
}

function Test-PairAlignment {
    Write-Log "Testing pair alignment..."
    
    $alignmentReport = Join-Path $OutputDir "alignment_report.json"
    
    if (-not (Test-Path $alignmentReport)) {
        Write-Log "  Alignment report not found" -Level "WARNING"
        return $true
    }
    
    try {
        $report = Get-Content $alignmentReport -Raw | ConvertFrom-Json
        
        $totalPairs = $report.total_pairs
        $alignedPairs = $report.aligned
        
        if ($totalPairs -eq 0) {
            Write-Log "  No pairs found in alignment report" -Level "WARNING"
            return $true
        }
        
        $alignmentRate = ($alignedPairs / $totalPairs) * 100
        Write-Log "  Aligned pairs: $alignedPairs/$totalPairs ($alignmentRate:F2%)"
        
        if ($alignmentRate -ge 95) {
            Write-Log "  ✓ Pair alignment acceptable (≥95%)" -Level "SUCCESS"
            return $true
        } else {
            Write-Log "  ✗ Pair alignment below threshold" -Level "ERROR"
            return $false
        }
        
    } catch {
        Write-Log "  ✗ Error reading alignment report: $_" -Level "ERROR"
        return $false
    }
}

function Test-QualityReports {
    Write-Log "Testing quality reports..."
    
    $requiredReports = @(
        "thermal_anomaly_report.json",
        "validation_report.txt",
        "qa_audit_report.txt"
    )
    
    $missingReports = @()
    foreach ($report in $requiredReports) {
        $reportPath = Join-Path $OutputDir $report
        if (Test-Path $reportPath) {
            Write-Log "  ✓ Found: $report"
        } else {
            Write-Log "  ✗ Missing: $report" -Level "WARNING"
            $missingReports += $report
        }
    }
    
    return $missingReports.Count -eq 0
}

function Generate-SummaryReport {
    param(
        [hashtable]$Results
    )
    
    $reportPath = Join-Path $OutputDir "verification_summary.txt"
    
    $reportLines = @(
        "NEODrone Dataset Verification Summary",
        "=" * 50,
        "Verification Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "Output Directory: $OutputDir",
        "",
        "Test Results:"
    )
    
    foreach ($test in $Results.Keys) {
        $result = $Results[$test]
        $status = if ($result) { "PASSED" } else { "FAILED" }
        $reportLines += "  $test : $status"
    }
    
    $passedCount = ($Results.Values | Where-Object { $_ -eq $true }).Count
    $totalCount = $Results.Count
    
    $reportLines += @(
        "",
        "Summary: $passedCount/$totalCount tests passed",
        ""
    )
    
    if ($passedCount -eq $totalCount) {
        $reportLines += "OVERALL STATUS: VERIFIED ✓"
    } else {
        $reportLines += "OVERALL STATUS: FAILED ✗"
    }
    
    $reportLines | Out-File -FilePath $reportPath -Encoding UTF8
    
    Write-Log "Verification summary saved to: $reportPath"
}

function Main {
    Write-Log "========================================"
    Write-Log "NEODrone Output Verification"
    Write-Log "========================================"
    Write-Log "Output Directory: $OutputDir"
    Write-Log "========================================"
    
    $results = @{}
    
    try {
        $results['Directory Structure'] = Test-DirectoryStructure
        $results['Frame Count'] = Test-FrameCount
        $results['Metadata Completeness'] = Test-MetadataCompleteness
        $results['Annotation Consistency'] = Test-AnnotationConsistency
        $results['Pair Alignment'] = Test-PairAlignment
        $results['Quality Reports'] = Test-QualityReports
        
        Generate-SummaryReport -Results $results
        
        $passedCount = ($results.Values | Where-Object { $_ -eq $true }).Count
        $totalCount = $results.Count
        
        Write-Log "========================================"
        Write-Log "Verification Complete!"
        Write-Log "Passed: $passedCount/$totalCount tests"
        
        if ($passedCount -eq $totalCount) {
            Write-Log "Overall Status: VERIFIED ✓" -Level "SUCCESS"
            exit 0
        } else {
            Write-Log "Overall Status: FAILED ✗" -Level "ERROR"
            exit 1
        }
        
    } catch {
        Write-Log "Verification failed with error: $_" -Level "ERROR"
        exit 1
    }
}

Main