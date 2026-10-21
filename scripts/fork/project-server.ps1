# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
    Manages a long-running Serena project server (`serena start-project-server`) on Windows.

.DESCRIPTION
    Actions:
      start      starts the server hidden and detached from the calling console, unless the port is already
                 served by a Serena project server
      stop       terminates the server's process tree; with -IncludeOrphans also orphaned language servers
      status     prints the server process, the heartbeat response and orphaned language servers
      install    registers a scheduled task that runs "start" with the given parameters at logon
      uninstall  removes the scheduled task

    The server process is identified via the port it listens on, not via a PID file, so that a reused PID can
    never cause an unrelated process to be terminated.
    The authentication secret (Serena v2) is read from serena_config.yml inside this process; it is never
    passed on a command line, logged or printed.

.EXAMPLE
    .\project-server.ps1 -Action install -Warm C:\repo\a,C:\repo\b -AllowedTool find_symbol,get_symbols_overview
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'status', 'install', 'uninstall')]
    [string]$Action,
    [string]$SerenaExe = 'C:\github\serenaFork\.venv\Scripts\serena.exe',
    [int]$Port = 24225,
    # names or root paths of projects to warm up at startup (Serena versions supporting --warm only)
    [string[]]$Warm = @(),
    # tools the server shall accept (Serena versions supporting --allowed-tool only)
    [string[]]$AllowedTool = @(),
    [switch]$IncludeOrphans,
    [string]$TaskName = 'SerenaProjectServer',
    [int]$StartTimeoutSeconds = 120,
    # time granted to a graceful shutdown (saving large symbol caches can take a while) before the process is killed
    [int]$ShutdownTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$LanguageServersMarker = '\.serena\language_servers\'

function Get-SerenaHome {
    if ($env:SERENA_HOME) { return $env:SERENA_HOME }
    return (Join-Path $env:USERPROFILE '.serena')
}

# -Warm a,b arrives as one string when the script is run via powershell.exe -File (as by the scheduled task)
function Split-ListArgument([string[]]$Values) {
    $items = @()
    foreach ($value in $Values) {
        foreach ($item in ($value -split ',')) {
            if ($item.Trim()) { $items += $item.Trim() }
        }
    }
    return , $items
}

function Get-AuthSecret {
    $configPath = Join-Path (Get-SerenaHome) 'serena_config.yml'
    if (-not (Test-Path $configPath)) { return $null }
    foreach ($line in Get-Content -LiteralPath $configPath -Encoding UTF8) {
        if ($line -match '^\s*auth_secret:\s*(.*?)\s*$') {
            $secret = $Matches[1].Trim('"', "'")
            if ($secret -and $secret -ne 'null') { return $secret }
        }
    }
    return $null
}

function Invoke-Heartbeat {
    $headers = @{}
    $secret = Get-AuthSecret
    if ($secret) { $headers['Authorization'] = "Bearer $secret" }
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/heartbeat" -Headers $headers -UseBasicParsing -TimeoutSec 5
        return [pscustomobject]@{ Code = [int]$response.StatusCode; Body = $response.Content }
    }
    catch [System.Net.WebException] {
        if ($_.Exception.Response) {
            return [pscustomobject]@{ Code = [int]$_.Exception.Response.StatusCode; Body = '' }
        }
        return [pscustomobject]@{ Code = 0; Body = $_.Exception.Message }
    }
}

function Get-ProcessTable {
    $table = @{}
    foreach ($process in Get-CimInstance Win32_Process) { $table[[int]$process.ProcessId] = $process }
    return $table
}

function Get-ListeningServer {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $connection) { return $null }

    # walk up to the outermost process of the server (uv's serena.exe launcher spawns the Python interpreters)
    $processes = Get-ProcessTable
    $process = $processes[[int]$connection.OwningProcess]
    if (-not $process) { return $null }
    $isServer = [bool]($process.CommandLine -and $process.CommandLine -like '*start-project-server*')
    $root = $process
    while ($isServer) {
        $parent = $processes[[int]$root.ParentProcessId]
        if (-not $parent -or -not $parent.CommandLine -or $parent.CommandLine -notlike '*start-project-server*') { break }
        if ($parent.CreationDate -gt $root.CreationDate) { break }
        $root = $parent
    }
    return [pscustomobject]@{ Listener = $process; Root = $root; IsServer = $isServer }
}

# language server processes whose parent is gone (or whose parent PID was reused by a younger process)
function Get-OrphanLanguageServers {
    $processes = Get-ProcessTable
    $orphans = @()
    foreach ($process in $processes.Values) {
        if (-not $process.CommandLine) { continue }
        $commandLine = $process.CommandLine.Replace('/', '\')
        if ($commandLine.IndexOf($LanguageServersMarker, [StringComparison]::OrdinalIgnoreCase) -lt 0) { continue }
        $parent = $processes[[int]$process.ParentProcessId]
        $parentIsLanguageServer = $parent -and $parent.CommandLine -and
            $parent.CommandLine.Replace('/', '\').IndexOf($LanguageServersMarker, [StringComparison]::OrdinalIgnoreCase) -ge 0
        if ($parentIsLanguageServer -and $parent.CreationDate -le $process.CreationDate) { continue }
        if (-not $parent -or $parent.CreationDate -gt $process.CreationDate) { $orphans += $process }
    }
    return , $orphans
}

function Format-Process($Process) {
    $commandLine = [string]$Process.CommandLine
    if ($commandLine.Length -gt 160) { $commandLine = $commandLine.Substring(0, 160) + '...' }
    return ('{0,6} {1:yyyy-MM-dd HH:mm:ss} {2}' -f $Process.ProcessId, $Process.CreationDate, $commandLine)
}

function Stop-ProcessTree([int]$ProcessId) {
    & taskkill.exe /T /F /PID $ProcessId | Out-Null
}

function Quote-Argument([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + $Value.Replace('"', '\"') + '"' }
    return $Value
}

function Get-ServerArguments {
    $arguments = @('start-project-server', '--port', [string]$Port)
    foreach ($project in (Split-ListArgument $Warm)) { $arguments += @('--warm', $project) }
    foreach ($tool in (Split-ListArgument $AllowedTool)) { $arguments += @('--allowed-tool', $tool) }
    return , $arguments
}

function Invoke-Start {
    $server = Get-ListeningServer
    if ($server) {
        if ($server.IsServer) {
            Write-Output "Serena project server already running on port $Port (PID $($server.Root.ProcessId))"
            return
        }
        throw "Port $Port is in use by another process: $(Format-Process $server.Listener)"
    }
    if (-not (Test-Path -LiteralPath $SerenaExe)) { throw "Serena executable not found: $SerenaExe" }

    # Start-Process redirection would let the server inherit the caller's console handles (keeping the
    # caller's pipes open for the server's lifetime), so cmd.exe redirects the output instead and the
    # process is started via ShellExecute in a hidden console of its own, detached from the calling terminal
    $logDir = Join-Path (Get-SerenaHome) 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $serverArguments = Get-ServerArguments
    $serverCommand = (@($SerenaExe) + $serverArguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $outLog = Join-Path $logDir 'project-server-console.out.log'
    $errLog = Join-Path $logDir 'project-server-console.err.log'
    $cmdArguments = '/d /s /c "' + $serverCommand + ' > "' + $outLog + '" 2> "' + $errLog + '""'
    $process = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdArguments -WindowStyle Hidden -PassThru
    # keeps the process handle, without which the exit code of an early exit is not available
    $null = $process.Handle

    # the port opens only once the agent is constructed; 401 also counts as up (server requires a secret)
    $deadline = (Get-Date).AddSeconds($StartTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) { throw "Serena project server exited with code $($process.ExitCode); see $logDir" }
        $heartbeat = Invoke-Heartbeat
        if ($heartbeat.Code -eq 200 -or $heartbeat.Code -eq 401) {
            Write-Output "Serena project server started on port $Port (PID $($process.Id), heartbeat $($heartbeat.Code))"
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Serena project server did not answer on port $Port within $StartTimeoutSeconds s; see $logDir"
}

# asks the server to shut down gracefully (stopping its language servers and saving their caches); servers without
# the /shutdown route (upstream Serena) answer 404 and are then terminated
function Request-GracefulShutdown([int]$ProcessId) {
    $headers = @{}
    $secret = Get-AuthSecret
    if ($secret) { $headers['Authorization'] = "Bearer $secret" }
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/shutdown" -Method Post -Headers $headers -UseBasicParsing -TimeoutSec 5 | Out-Null
    }
    catch {
        return $false
    }
    $deadline = (Get-Date).AddSeconds($ShutdownTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Invoke-Stop {
    $server = Get-ListeningServer
    if ($server -and $server.IsServer) {
        if (Request-GracefulShutdown $server.Listener.ProcessId) {
            Write-Output "Shut down Serena project server gracefully (PID $($server.Root.ProcessId))"
        }
        else {
            Stop-ProcessTree $server.Root.ProcessId
            Write-Output "Terminated Serena project server (PID $($server.Root.ProcessId))"
        }
    }
    elseif ($server) {
        throw "Port $Port is in use by a process that is not a Serena project server: $(Format-Process $server.Listener)"
    }
    else {
        Write-Output "No Serena project server is listening on port $Port"
    }

    if ($IncludeOrphans) {
        # the server's own language servers become orphans only after their parents are gone
        Start-Sleep -Seconds 1
        foreach ($orphan in (Get-OrphanLanguageServers)) {
            Stop-ProcessTree $orphan.ProcessId
            Write-Output "Stopped orphaned language server: $(Format-Process $orphan)"
        }
    }
}

function Invoke-Status {
    $server = Get-ListeningServer
    if (-not $server) {
        Write-Output "Port ${Port}: no listener"
    }
    elseif (-not $server.IsServer) {
        Write-Output "Port ${Port}: in use by another process: $(Format-Process $server.Listener)"
    }
    else {
        Write-Output "Port ${Port}: Serena project server"
        Write-Output "  root:     $(Format-Process $server.Root)"
        Write-Output "  listener: $(Format-Process $server.Listener)"
        $heartbeat = Invoke-Heartbeat
        $meaning = switch ($heartbeat.Code) { 200 { 'alive' } 401 { 'alive, secret rejected' } default { 'not answering' } }
        Write-Output "  heartbeat: $($heartbeat.Code) ($meaning) $($heartbeat.Body)"
    }

    $orphans = Get-OrphanLanguageServers
    Write-Output "Orphaned language servers: $($orphans.Count)"
    foreach ($orphan in $orphans) { Write-Output "  $(Format-Process $orphan)" }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Output "Scheduled task '$TaskName': $($task.State); $($task.Actions[0].Execute) $($task.Actions[0].Arguments)"
    }
    else {
        Write-Output "Scheduled task '$TaskName': not installed"
    }
}

function Invoke-Install {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $taskArguments = @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
        '-File', (Quote-Argument $PSCommandPath), '-Action', 'start',
        '-SerenaExe', (Quote-Argument $SerenaExe), '-Port', [string]$Port
    )
    $warmList = Split-ListArgument $Warm
    if ($warmList.Count) { $taskArguments += @('-Warm', (Quote-Argument ($warmList -join ','))) }
    $toolList = Split-ListArgument $AllowedTool
    if ($toolList.Count) { $taskArguments += @('-AllowedTool', (Quote-Argument ($toolList -join ','))) }

    $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ($taskArguments -join ' ')
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger -Principal $principal `
        -Settings $settings -Description 'Starts the Serena project server at logon' -Force | Out-Null
    Write-Output "Installed scheduled task '$TaskName': powershell.exe $($taskArguments -join ' ')"
}

function Invoke-Uninstall {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Removed scheduled task '$TaskName'"
    }
    else {
        Write-Output "Scheduled task '$TaskName' is not installed"
    }
}

switch ($Action) {
    'start' { Invoke-Start }
    'stop' { Invoke-Stop }
    'status' { Invoke-Status }
    'install' { Invoke-Install }
    'uninstall' { Invoke-Uninstall }
}
