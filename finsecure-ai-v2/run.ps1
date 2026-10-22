param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ArgsList
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
python run.py @ArgsList
