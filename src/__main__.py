if __name__ == "__main__":
    try:
        from .main import main
        main()
    except (KeyboardInterrupt):
        print("[ERROR]: Keyboard interruption fron the user")
    except (Exception) as e:
        print(e)