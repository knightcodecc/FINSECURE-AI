param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ArgsList
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
if (Test-Path "finsecure-ai-v2\run.py") {
    Set-Location "finsecure-ai-v2"
}
python run.py @ArgsList
