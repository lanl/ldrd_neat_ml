import pandas as pd


def main():
    df = pd.read_excel("data/mihee_peo_dextran_phase_map_experimental.xlsx")
    print("df:\n", df)


if __name__ == "__main__":
    main()
