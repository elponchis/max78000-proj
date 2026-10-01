# MAX78000FTHR 시리얼 읽기 — **Windows PowerShell 에서 돌린다.**
#
# 왜 이 스크립트인가: Windows 8 이후 기본 시리얼 터미널이 없고, Tera Term /
# PuTTY 를 따로 깔지 않아도 .NET 의 SerialPort 로 충분하다.
# **보드 브링업·실험 로그 수집에 그대로 쓴다.**
#
# 사용 (Windows PowerShell — WSL 아님):
#   powershell -ExecutionPolicy Bypass -File scripts\serial_read.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\serial_read.ps1 -Port COM3 -Seconds 20
#   powershell -ExecutionPolicy Bypass -File scripts\serial_read.ps1 -OutFile log.txt
#
# ⚠️ 보율 기본값 115200 은 MSDK FTHR_RevA BSP 의 `CONSOLE_BAUD` 다
#    (Libraries/Boards/MAX78000/FTHR_RevA/Include/board.h:48).
param(
    [string]$Port = 'COM3',
    [int]$Baud = 115200,
    [int]$Seconds = 10,
    [string]$OutFile = '',
    # 깨진 바이트를 '.' 로 바꿔 보여 준다 (보율 불일치 진단용)
    [switch]$Sanitize
)

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, 'None', 8, 'One'
# ⚠️ 흐름제어는 반드시 None. DAPLink CDC 는 RTS/CTS 를 쓰지 않는다.
$sp.Handshake = 'None'
$sp.DtrEnable = $true
$sp.RtsEnable = $true
$sp.ReadTimeout = 500

try {
    $sp.Open()
} catch {
    Write-Output ("[에러] {0} 열기 실패: {1}" -f $Port, $_.Exception.Message)
    Write-Output "  - 다른 프로그램이 포트를 잡고 있지 않은지 확인 (Tera Term 등)"
    Write-Output "  - 장치 관리자 > 포트(COM & LPT) 에서 번호 재확인"
    exit 1
}

Write-Output ("=== {0} @ {1} baud, {2}초 수신" -f $Port, $Baud, $Seconds)
$buf = New-Object System.Text.StringBuilder
$sw = [Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
    try {
        $chunk = $sp.ReadExisting()
        if ($chunk.Length -gt 0) { [void]$buf.Append($chunk) }
    } catch {}
    Start-Sleep -Milliseconds 100
}
if ($sp.IsOpen) { $sp.Close() }
$sp.Dispose()

$text = $buf.ToString()
if ($text.Length -eq 0) {
    Write-Output "(수신 0 바이트)"
    Write-Output "  - 펌웨어를 플래싱했는지 확인 (.bin 을 DAPLINK 드라이브에 드래그)"
    Write-Output "  - 보드의 RST 버튼을 한 번 누른 뒤 다시 실행"
    exit 2
}

# 출력가능 문자 비율 — 보율이 맞으면 거의 100% 다
$chars = $text.ToCharArray()
$ok = ($chars | Where-Object { [int]$_ -ge 32 -and [int]$_ -lt 127 }).Count
$ws = ($chars | Where-Object { $_ -eq "`n" -or $_ -eq "`r" -or $_ -eq "`t" }).Count
$ratio = [math]::Round(100 * ($ok + $ws) / $chars.Count)
Write-Output ("{0} 바이트, 출력가능 {1}%" -f $chars.Count, $ratio)
if ($ratio -lt 90) {
    Write-Output "  [주의] 출력가능 비율이 낮다 - **보율 불일치**일 가능성이 크다"
}
Write-Output "--------"
if ($Sanitize) { Write-Output ($text -replace "[^\x20-\x7E\r\n\t]", '.') }
else { Write-Output $text }

if ($OutFile -ne '') {
    $text | Out-File -FilePath $OutFile -Encoding utf8
    Write-Output ("--------")
    Write-Output ("저장: {0}" -f $OutFile)
}
