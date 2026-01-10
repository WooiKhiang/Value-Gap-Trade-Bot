from .monitor_5m import loop_forever

if __name__ == "__main__":
    # Run every 60 seconds; confirmation logic only looks at last ~9 closed 5m candles anyway
    loop_forever(poll_seconds=60)
