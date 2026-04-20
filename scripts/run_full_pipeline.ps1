#!/usr/bin/env pwsh

<#
.SYNOPSIS
    NEODrone Dataset Processing Pipeline - One-Click Reproduction Script

.DESCRIPTION
    This script automates the complete NEODrone dataset processing pipeline including:
    1. Data extraction from videos
    2. Multi-modal synchronization
    3. Metadata parsing
    4. Data cleaning and quality control
    5. Annotation training and inference
    6. Metadata validation and QA audit

.PARAMETER InputDir
    Input directory containing raw videos

.PARAMETER OutputDir
    Output directory for processed data

.PARAMETER ConfigDir
    Directory containing configuration files

.PARAMETER SkipExtraction
    Skip data extraction step

.PARAMETER SkipCleaning
    Skip data cleaning step

.PARAMETER SkipAnnotation
    Skip annotation step

.PARAMETER SkipValidation
    Skip validation step

.EXAMPLE
    .\run_full_pipeline.ps1 -InputDir "raw_videos" -OutputDir "processed_data"

.EXAMPLE
    .\run_full_pipeline.ps1 -InputDir "raw_videos" -OutputDir "processed_data" -SkipExtraction
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [string]$ConfigDir = "configs",

    [switch]$SkipExtraction = $false,

    [switch]$SkipCleaning = $false,

    [switch]$SkipAnnotation = $false,

    [switch]$SkipValidation = $false
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
        default { "White" }
    }
    
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-Dependencies {
    Write-Log "Checking dependencies..."
    
    $requiredModules = @("opencv-python", "numpy", "pandas", "torch", "ultralytics")
    $missingModules = @()
    
    foreach ($module in $requiredModules) {
        try {
            $result = python -c "import $module; print('OK')" 2>&1
            if ($result -match "OK") {
                Write-Log "  ✓ $module is installed"
            } else {
                Write-Log "  ✗ $module is not properly installed" -Level "WARNING"
                $missingModules += $module
            }
        } catch {
            Write-Log "  ✗ $module is not installed" -Level "WARNING"
            $missingModules += $module
        }
    }
    
    if ($missingModules.Count -gt 0) {
        Write-Log "Missing dependencies: $($missingModules -join ', ')" -Level "ERROR"
        Write-Log "Please install missing dependencies using: pip install -r requirements_*.txt" -Level "ERROR"
        exit 1
    }
    
    Write-Log "All dependencies are installed!"
}

function Invoke-DataExtraction {
    Write-Log "Starting data extraction..."
    
    $framesDir = Join-Path $OutputDir "frames"
    $configFile = Join-Path $ConfigDir "extraction_config.yaml"
    
    Write-Log "Extracting frames from videos..."
    python src/extraction/extract_frames.py `
        --input $InputDir `
        --output $framesDir `
        --config $configFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Data extraction failed!" -Level "ERROR"
        exit 1
    }
    
    Write-Log "Data extraction completed!"
}

function Invoke-MultiModalSync {
    Write-Log "Starting multi-modal synchronization..."
    
    $visibleDir = Join-Path $OutputDir "frames/visible"
    $thermalDir = Join-Path $OutputDir "frames/thermal"
    $syncedDir = Join-Path $OutputDir "synced"
    
    if ((Test-Path $visibleDir) -and (Test-Path $thermalDir)) {
        Write-Log "Synchronizing visible and thermal frames..."
        python src/extraction/sync_extract.py `
            --visible $visibleDir `
            --thermal $thermalDir `
            --output $syncedDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Multi-modal synchronization failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Multi-modal synchronization completed!"
    } else {
        Write-Log "Skipping multi-modal sync (visible or thermal frames not found)" -Level "WARNING"
    }
}

function Invoke-MetadataParsing {
    Write-Log "Starting metadata parsing..."
    
    $sdkLogsDir = Join-Path $InputDir "sdk_logs"
    $exifDataDir = Join-Path $OutputDir "frames"
    $metadataFile = Join-Path $OutputDir "metadata.xlsx"
    $vocabFile = Join-Path $ConfigDir "metadata_vocab.json"
    
    if (Test-Path $sdkLogsDir) {
        Write-Log "Parsing metadata from SDK logs and EXIF data..."
        python src/extraction/parse_metadata.py `
            --sdk_logs $sdkLogsDir `
            --exif_data $exifDataDir `
            --output $metadataFile `
            --vocab $vocabFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Metadata parsing failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Metadata parsing completed!"
    } else {
        Write-Log "Skipping metadata parsing (SDK logs directory not found)" -Level "WARNING"
    }
}

function Invoke-DataCleaning {
    Write-Log "Starting data cleaning..."
    
    $inputDir = Join-Path $OutputDir "synced"
    $cleanedDir = Join-Path $OutputDir "cleaned"
    $configFile = Join-Path $ConfigDir "cleaning_thresholds.yaml"
    
    if (Test-Path $inputDir) {
        Write-Log "Cleaning and filtering image pairs..."
        python src/cleaning/clean_pipeline.py `
            --input $inputDir `
            --output $cleanedDir `
            --config $configFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Data cleaning failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Data cleaning completed!"
    } else {
        Write-Log "Skipping data cleaning (synced directory not found)" -Level "WARNING"
    }
}

function Invoke-ThermalAnomalyDetection {
    Write-Log "Starting thermal anomaly detection..."
    
    $cleanedDir = Join-Path $OutputDir "cleaned"
    $anomalyReport = Join-Path $OutputDir "thermal_anomaly_report.json"
    
    if (Test-Path $cleanedDir) {
        Write-Log "Detecting thermal anomalies..."
        python src/cleaning/thermal_anomaly_detect.py `
            --input $cleanedDir `
            --output $anomalyReport
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Thermal anomaly detection failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Thermal anomaly detection completed!"
    } else {
        Write-Log "Skipping thermal anomaly detection (cleaned directory not found)" -Level "WARNING"
    }
}

function Invoke-AlignmentCheck {
    Write-Log "Starting alignment check..."
    
    $cleanedDir = Join-Path $OutputDir "cleaned"
    $alignmentReport = Join-Path $OutputDir "alignment_report.json"
    
    if (Test-Path $cleanedDir) {
        Write-Log "Checking alignment between visible and thermal images..."
        python src/cleaning/align_check.py `
            --visible (Join-Path $cleanedDir "visible") `
            --thermal (Join-Path $cleanedDir "thermal") `
            --output $alignmentReport
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Alignment check failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Alignment check completed!"
    } else {
        Write-Log "Skipping alignment check (cleaned directory not found)" -Level "WARNING"
    }
}

function Invoke-AnnotationTraining {
    Write-Log "Starting annotation model training..."
    
    $configFile = Join-Path $ConfigDir "annotation_config.yaml"
    
    if (Test-Path $configFile) {
        Write-Log "Training RT-DETR model..."
        python src/annotation/train_rtdetr.py `
            --config $configFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Annotation training failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Annotation training completed!"
    } else {
        Write-Log "Skipping annotation training (config file not found)" -Level "WARNING"
    }
}

function Invoke-AnnotationInference {
    Write-Log "Starting annotation inference..."
    
    $weightsFile = "weights/rtdetr_neodrone.pt"
    $cleanedDir = Join-Path $OutputDir "cleaned"
    $annotationsDir = Join-Path $OutputDir "annotations"
    $configFile = Join-Path $ConfigDir "annotation_config.yaml"
    
    if ((Test-Path $weightsFile) -and (Test-Path $cleanedDir)) {
        Write-Log "Running annotation inference..."
        python src/annotation/inference_rtdetr.py `
            --weights $weightsFile `
            --input $cleanedDir `
            --output $annotationsDir `
            --config $configFile `
            --format yolo
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Annotation inference failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Annotation inference completed!"
    } else {
        Write-Log "Skipping annotation inference (weights or cleaned directory not found)" -Level "WARNING"
    }
}

function Invoke-MetadataValidation {
    Write-Log "Starting metadata validation..."
    
    $metadataFile = Join-Path $OutputDir "metadata.xlsx"
    $validationDir = Join-Path $OutputDir "validation"
    $vocabFile = Join-Path $ConfigDir "metadata_vocab.json"
    
    if (Test-Path $metadataFile) {
        Write-Log "Validating metadata schema..."
        python src/metadata/validate_schema.py `
            --metadata $metadataFile `
            --vocab $vocabFile `
            --output $validationDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Metadata validation failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Metadata validation completed!"
    } else {
        Write-Log "Skipping metadata validation (metadata file not found)" -Level "WARNING"
    }
}

function Invoke-QAAudit {
    Write-Log "Starting QA audit..."
    
    $metadataFile = Join-Path $OutputDir "metadata.xlsx"
    $qaDir = Join-Path $OutputDir "qa_audit"
    
    if (Test-Path $metadataFile) {
        Write-Log "Running QA audit..."
        python src/metadata/qa_audit_log.py `
            --metadata $metadataFile `
            --output $qaDir `
            --sample 5.0
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "QA audit failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "QA audit completed!"
    } else {
        Write-Log "Skipping QA audit (metadata file not found)" -Level "WARNING"
    }
}

function Invoke-OutputVerification {
    Write-Log "Starting output verification..."
    
    $verifyScript = "scripts/verify_output.ps1"
    if (Test-Path $verifyScript) {
        Write-Log "Verifying output quality..."
        & $verifyScript -OutputDir $OutputDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Output verification failed!" -Level "ERROR"
            exit 1
        }
        
        Write-Log "Output verification completed!"
    } else {
        Write-Log "Skipping output verification (verify script not found)" -Level "WARNING"
    }
}

function Main {
    Write-Log "========================================"
    Write-Log "NEODrone Dataset Processing Pipeline"
    Write-Log "========================================"
    Write-Log "Input Directory: $InputDir"
    Write-Log "Output Directory: $OutputDir"
    Write-Log "Config Directory: $ConfigDir"
    Write-Log "========================================"
    
    $startTime = Get-Date
    
    try {
        Test-Dependencies
        
        if (-not $SkipExtraction) {
            Invoke-DataExtraction
            Invoke-MultiModalSync
            Invoke-MetadataParsing
        }
        
        if (-not $SkipCleaning) {
            Invoke-DataCleaning
            Invoke-ThermalAnomalyDetection
            Invoke-AlignmentCheck
        }
        
        if (-not $SkipAnnotation) {
            Invoke-AnnotationTraining
            Invoke-AnnotationInference
        }
        
        if (-not $SkipValidation) {
            Invoke-MetadataValidation
            Invoke-QAAudit
        }
        
        Invoke-OutputVerification
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Log "========================================"
        Write-Log "Pipeline completed successfully!"
        Write-Log "Total duration: $($duration.ToString('hh\:mm\:ss'))"
        Write-Log "========================================"
        
    } catch {
        Write-Log "Pipeline failed with error: $_" -Level "ERROR"
        exit 1
    }
}

Main