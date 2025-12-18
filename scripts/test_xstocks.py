#!/usr/bin/env python3
"""
Test which xStocks are available on Kraken WebSocket v2.
"""

import asyncio
import json
import ssl
import sys

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

try:
    import certifi
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "certifi", "-q"])
    import certifi


# All potential xStocks to test
XSTOCKS_TO_TEST = [
    # Tech Giants
    "AAPLx/USD", "MSFTx/USD", "GOOGLx/USD", "AMZNx/USD", "METAx/USD",
    "NVDAx/USD", "TSLAx/USD", "AMDx/USD", "INTCx/USD", "AVGOx/USD",
    # Streaming/Entertainment
    "NFLXx/USD", "DISx/USD", "WBDx/USD",
    # Fintech/Crypto-related
    "COINx/USD", "MSTRx/USD", "SQx/USD", "PYPLx/USD", "HOODx/USD",
    # Software/Cloud
    "CRMx/USD", "ORCLx/USD", "ADOBEx/USD", "SNOWx/USD", "PLTRx/USD",
    # E-commerce/Consumer
    "SHOPx/USD", "BABAx/USD", "JDx/USD", "PINSx/USD",
    # Aerospace/Defense
    "BAx/USD", "LMTx/USD", "RTXx/USD",
    # Automotive
    "Fx/USD", "GMx/USD", "RIVNx/USD", "LCIDx/USD",
    # Healthcare/Pharma
    "PFEx/USD", "MRNAx/USD", "JNJx/USD",
    # Banking/Finance
    "JPMx/USD", "BACx/USD", "GSx/USD", "MSx/USD",
    # ETFs/Indices
    "SPYx/USD", "QQQx/USD", "TQQQx/USD", "SQQQx/USD",
    "DIAx/USD", "IWMx/USD", "SOXXx/USD", "ARKKx/USD",
    "VOOx/USD", "VTIx/USD", "GLDx/USD", "SLVx/USD",
    # Meme stocks
    "GMEx/USD", "AMCx/USD", "BBx/USD",
    # Additional potential stocks
    "UBERx/USD", "LYFTx/USD", "SNAPx/USD", "TWTRx/USD",
    "ZMx/USD", "DOCUx/USD", "CROWDx/USD", "NETx/USD",
    "DBSx/USD", "Ux/USD", "ROKUx/USD", "SPOTx/USD",
]


async def test_xstocks(timeout_seconds: int = 15):
    """Test which xStocks are available on Kraken WebSocket."""
    uri = "wss://ws.kraken.com/v2"
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Track which pairs receive data
    pairs_with_data = {}
    errors = []

    print(f"Conectando a {uri}...")
    print(f"Probando {len(XSTOCKS_TO_TEST)} xStocks...")
    print("=" * 60)

    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            # Subscribe to ticker for all pairs
            sub_msg = {
                "method": "subscribe",
                "params": {
                    "channel": "ticker",
                    "symbol": XSTOCKS_TO_TEST
                }
            }
            await ws.send(json.dumps(sub_msg))

            print(f"Esperando datos por {timeout_seconds} segundos...\n")

            # Listen for responses
            end_time = asyncio.get_event_loop().time() + timeout_seconds
            while asyncio.get_event_loop().time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1)
                    data = json.loads(msg)

                    # Check for subscription errors
                    if data.get("error"):
                        errors.append(data)

                    # Check for ticker data
                    if data.get("channel") == "ticker" and "data" in data:
                        for item in data["data"]:
                            symbol = item.get("symbol", "")
                            if symbol and symbol not in pairs_with_data:
                                last_price = item.get("last", "N/A")
                                pairs_with_data[symbol] = last_price
                                print(f"  ✓ {symbol}: ${last_price}")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Connection error: {e}")
        return

    # Print summary
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    working = sorted(pairs_with_data.keys())
    not_working = sorted(set(XSTOCKS_TO_TEST) - set(working))

    print(f"\n✓ xStocks CON datos ({len(working)}):")
    print("-" * 40)
    for pair in working:
        print(f"  {pair}: ${pairs_with_data[pair]}")

    if not_working:
        print(f"\n✗ xStocks SIN datos ({len(not_working)}):")
        print("-" * 40)
        for pair in not_working:
            print(f"  {pair}")

    # Print config-ready format
    print("\n" + "=" * 60)
    print("FORMATO PARA config/default.toml:")
    print("=" * 60)
    print("stock_pairs = [")
    for i, pair in enumerate(working):
        comma = "," if i < len(working) - 1 else ""
        print(f'    "{pair}"{comma}')
    print("]")

    return working, not_working


if __name__ == "__main__":
    print("=" * 60)
    print("KRAKEN xSTOCKS AVAILABILITY TEST")
    print("=" * 60)
    print()

    # Note about market hours
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    et_hour = (now.hour - 5) % 24  # Rough ET conversion
    print(f"Hora UTC: {now.strftime('%H:%M')}")
    print(f"Hora ET aprox: {et_hour:02d}:{now.minute:02d}")
    print(f"Mercado NYSE: 9:30 AM - 4:00 PM ET")

    if et_hour < 9 or (et_hour == 9 and now.minute < 30) or et_hour >= 16:
        print("\n⚠️  NOTA: El mercado puede estar cerrado.")
        print("    Los xStocks solo envían datos durante horario de mercado.")
    print()

    asyncio.run(test_xstocks(timeout_seconds=15))
