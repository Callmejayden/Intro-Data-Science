import pandas as pd
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from io import StringIO
import re


class EuropeanFootballAnalysis:
    def __init__(self, filename = "Player_Stats.html"):

        # folder containing this file
        base_dir = Path(__file__).resolve().parent   

        # full path to HTML
        self.file = base_dir / filename

        # Container for Raw data
        self.raw_data = None 
        
        # Container for cleaned data
        self.cleaned_data = None
        
        
        # Mapping dictionaries      
        self.position_map = {
            'GK': 'Goalkeeper',
            'DF': 'Defender', 
            'MF': 'Midfielder',
            'FW': 'Forward'
        }
        
        self.colors = {
            'Premier League': '#38003c',
            'La Liga': '#eda132',
            'Bundesliga': '#d3010c',
            'Serie A': '#008fd8',
            'Ligue 1': '#091c3e'
        }

        self.team_colors = {
          # Premier League
          'Manchester City': '#6CABDD',
          'Arsenal': '#EF0107',
          'Liverpool': '#C8102E',
          'Manchester United': '#DA291C',
          'Tottenham': '#FFFFFF',
          'Chelsea': '#034694',
          'Newcastle United': '#241F20',
          'Brighton': '#0057B8',
          'Aston Villa': '#95BFE5',
          'West Ham': '#7A263A',
          
          # La Liga  
          'Real Madrid': '#FEBE10',
          'Barcelona': '#A50044',
          'Atletico Madrid': '#CB3524',
          'Sevilla': '#FFFFFF',
          'Real Betis': '#00954C',
          'Villarreal': '#FFE667',
          'Real Sociedad': '#00529F',
          'Athletic Club': '#EE2523',
          
          # Serie A
          'Inter': '#010E80',
          'Milan': '#FB090B',
          'Juventus': '#000000',
          'Napoli': '#12A0D7',
          'Roma': '#8E1F2F',
          'Lazio': '#FFFFFF',
          'Atalanta': '#1D5EA8',
          'Fiorentina': '#482E92',
          
          # Bundesliga
          'Bayern Munich': '#0066B3',
          'Dortmund': '#FDE100',
          'RB Leipzig': '#DD0741',
          'Leverkusen': '#E32219',
          'Eintracht Frankfurt': '#E21A23',
          'Wolfsburg': '#65FF00',
          'Freiburg': '#E30613',
          "M'Gladbach": '#000000',
          
          # Ligue 1
          'Paris S-G': '#004170',
          'Marseille': '#00B9F1',
          'Lyon': '#D61C28',
          'Monaco': '#E30613',
          'Lille': '#E21A23',
          'Rennes': '#E21A23',
          'Lens': '#FFDC00',
          'Nice': '#000000'
      }

    def scrape(self):
        """
        Task 1: Read data from the html file and extraction table.
        populates self.raw_data as a pandas DataFrame.
        """
        # open the HTML file and parse it using BeautifulSoup with the lxml parser
        with open(self.file, "r", encoding="utf-8") as fp:
            soup = BeautifulSoup(fp,'lxml')

        # Locate the Standard Stats table by its HTML id
        table = soup.find("table", id="stats_standard")

        # convert the HTML table into a pandas and store it as raw_data
        self.raw_data = pd.read_html(StringIO(str(table)))[0]

        def flatten_col(col):
            if isinstance(col, tuple):
                parts = [str(p).strip() for p in col if 'Unnamed' not in str(p)]
                return ' '.join(parts).strip() or col[-1]
            return col

        self.raw_data.columns = [flatten_col(c) for c in self.raw_data.columns]

        # print to test !
        #print(self.raw_data.shape)
        #print(self.raw_data.head(10))

        pass

    def clean_data(self):

        """
        Task 2 & 3: Clean data.
        - Remove special characters from data
        - Convert columns with commas to numeric data
        - Parse Nation: extract 3-letter code from flag images or text
        - Parse Position: extract primary position and add a new column to data
        - Convert Columns to numbers based on the type of data
        - Populates self.cleaned_data as a pandas DataFrame.

        -Also Maps Team Colors and Position Full Names for later use in visualizations.
        """


        self.cleaned_data = self.raw_data.copy()
        self.cleaned_data.columns = [str(c).strip() for c in self.cleaned_data.columns]

        # drop rows where all elements are NaN
        self.cleaned_data.dropna(how='all', inplace=True)  
        self.cleaned_data.reset_index(drop=True, inplace=True)

        # drop matches col, not useful for analysis
        self.cleaned_data.drop("Matches", axis=1, inplace=True)

        # Remove special characters (keeps letters, numbers, underscores and whitespace)
        self.cleaned_data = self.cleaned_data.replace(to_replace=r'[^\w\s]', value='', regex=True)


        # Strip whitespace from common string columns
        cols_string = ["Squad", "League", "Nation", "Pos", "Player", "Team"]
        for c in cols_string:
            if c in self.cleaned_data.columns and self.cleaned_data[c].dtype == object:
                self.cleaned_data[c] = self.cleaned_data[c].str.strip()
        # remove repeated header rows
        if "Playing Time Min" in self.cleaned_data.columns:
            self.cleaned_data = self.cleaned_data[
                self.cleaned_data["Playing Time Min"].astype(str).str.strip() != "Min"
            ].copy()

        if "Playing Time MP" in self.cleaned_data.columns:
            self.cleaned_data = self.cleaned_data[
                self.cleaned_data["Playing Time MP"].astype(str).str.strip() != "MP"
            ].copy()


        # convert numeric columns to numbers
        cols_numeric = [
            "Age",
            "Playing Time MP", "Playing Time Min",
            "Performance Gls", "Performance Ast",
            "Performance CrdY", "Performance CrdR",
            "Per 90 Minutes Gls", "Per 90 Minutes Ast", "Per 90 Minutes G+A",
        ]

        for c in cols_numeric:
            if c in self.cleaned_data.columns:
                self.cleaned_data[c] = (
                    self.cleaned_data[c]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                self.cleaned_data[c] = pd.to_numeric(self.cleaned_data[c], errors="coerce")


        # Populate primary position column using helper
        if "Pos" in self.cleaned_data.columns:
            self.cleaned_data["Primary_Pos"] = self.cleaned_data["Pos"].apply(self._extract_primary_position)
        else:
            self.cleaned_data["Primary_Pos"] = np.nan

        # Map primary position to full name
        self.cleaned_data["Primary_Pos_Full"] = self.cleaned_data["Primary_Pos"].map(self.position_map)

        # Map team to color
        self.cleaned_data["Team_Color"] = self.cleaned_data["Squad"].map(self.team_colors)

        # Nation 3 letter code
        if "Nation" in self.cleaned_data.columns:
            self.cleaned_data["Nation"] = self.cleaned_data["Nation"].astype(str).str.extract(r"([A-Z]{3})", expand=False)

        # extract League from comp
        if "Comp" in self.cleaned_data.columns:
            self.cleaned_data["League"] = self.cleaned_data["Comp"].astype(str).str.split(" ", n=1).str[-1].str.strip()

        # NaN -> 0
        num_cols = self.cleaned_data.select_dtypes(include=["number"]).columns
        self.cleaned_data[num_cols] = self.cleaned_data[num_cols].fillna(0)
        # print(self.cleaned_data[num_cols].isna().sum()) # check NaN for task2

        pass


    def _extract_primary_position(self, pos_str):
        """
        Helper for Task 3: Extract primary position from codes like 'FWMF' or 'DF,MF'.
        Returns one of 'GK','DF','MF','FW' when possible, otherwise a short token.
        """

        codes = ['GK', 'DF', 'MF', 'FW']

        if pd.isna(pos_str):
            return np.nan
        string = str(pos_str).strip()

        # split on comma, slash, or whitespace and take first token
        token = re.split(r'[,/\s]+', string)[0].upper()

        if token in codes:
            return token

        for code in codes:
            if token.startswith(code):
                return code
        
        # fallback: return first two characters
        return token[:2]


    def add_derived_metrics(self):
        """
        Task 4: Add Minutes_per_Game, and Goal_Contribution_Rate. The derived metrics should be added as new columns to self.cleaned_data.
        """
        df = self.cleaned_data.copy()

        mp_col = "Playing Time MP"
        min_col = "Playing Time Min"
        gls_col = "Performance Gls"
        ast_col = "Performance Ast"
        nineties_col = "Playing Time 90s"

        # Minutes per game
        df["Minutes_per_Game"] = np.where(df[mp_col] > 0, df[min_col] / df[mp_col], 0)

        # create 90s column if missing
        if nineties_col not in df.columns:
            df[nineties_col] = np.where(df[min_col] > 0, df[min_col] / 90.0, 0)
        else:
            df[nineties_col] = pd.to_numeric(df[nineties_col], errors="coerce").fillna(0)
            df[nineties_col] = np.where(df[nineties_col] > 0, df[nineties_col], df[min_col] / 90.0)

        # Goal contribution rate (G+A per 90)
        df["Goal_Contribution_Rate"] = np.where(
            df[nineties_col] > 0,
            (df[gls_col] + df[ast_col]) / df[nineties_col],
            0
        )

        self.cleaned_data = df
        return df
        pass

    def find_top_scorer(self):
        """
        Task 5: Find the player(s) with the highest number of goals. Return player name, team, goals, league.
        """
        df = self.cleaned_data.copy()

        player_col = "Player"
        team_col = "Squad"
        goals_col = "Performance Gls"
        league_col = "League"

        # if goals -> string, convert to num
        df[goals_col] = pd.to_numeric(df[goals_col], errors='coerce').fillna(0)

        max_goals = df[goals_col].max()

        top = df.loc[df[goals_col] == max_goals, [player_col, team_col, goals_col, league_col]].copy()
        top = top.sort_values([league_col, team_col, player_col]).reset_index(drop=True)

        return top        
        pass

    def find_playmaker(self):
        """
        Task 6: Find the player(s) with the highest number of assists. Return player name, team, assists, league.
        """
        df = self.cleaned_data.copy()

        player_col = "Player"
        team_col = "Squad"
        assists_col = "Performance Ast"
        league_col = "League"

        # convert to num
        df[assists_col] = pd.to_numeric(df[assists_col], errors='coerce').fillna(0)
        max_assists = df[assists_col].max()

        playmakers = df.loc[df[assists_col] == max_assists, [player_col, team_col, assists_col, league_col]].copy()
        return playmakers.reset_index(drop=True)
    

    def find_ironman(self):
        """
        Task 7: Find the player(s) with the highest number of minutes played. Return player name, team, minutes played, matches played.
        """
        df = self.cleaned_data.copy()

        player_col = "Player"
        team_col = "Squad"
        minutes_col = "Playing Time Min"
        matches_col = "Playing Time MP"

        # convert num value
        df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0)
        df[matches_col] = pd.to_numeric(df[matches_col], errors="coerce").fillna(0)

        # find max minute
        max_minutes = df[minutes_col].max()
        #select players who have maximum minute
        ironman = df.loc[
            df[minutes_col] == max_minutes,
            [player_col, team_col, minutes_col, matches_col]
        ].copy()

        ironman = ironman.sort_values(
            [team_col, player_col]
        ).reset_index(drop=True)

        return ironman

    def find_efficient_striker(self):
        """
        Task 8: Find the forward(s) who is most efficent (most goals per minutes played) and has played more than 1000 minutes. Return player name, team, goals, minutes played, goals per minute.
        """
        df = self.cleaned_data.copy()
        player_col = "Player"
        team_col = "Squad"
        goals_col = "Performance Gls"
        min_col = "Playing Time Min"
        pos_col = "Primary_Pos"

        # goals and min -> num
        df[goals_col] = pd.to_numeric(df[goals_col], errors='coerce').fillna(0)
        df[min_col] = pd.to_numeric(df[min_col], errors='coerce').fillna(0)

        # filter only forwards with >1000 min
        fw = df[(df[pos_col] == 'FW') & (df[min_col] > 1000)].copy()

        # no qualifying forwards exist -> return empty
        if fw.empty:
            return pd.DataFrame(columns=[player_col, team_col, goals_col, min_col, "Goals_per_Minute"])
        
        # calculate 
        fw["Goals_per_Minute"] = np.where(fw[min_col] > 0, fw[goals_col] / fw[min_col], 0)
        # find max
        max_gpm = fw["Goals_per_Minute"].max()

        # select player with hightest goal per min
        best = fw.loc[fw["Goals_per_Minute"] == max_gpm, [player_col, team_col, goals_col, min_col, "Goals_per_Minute"]].copy()
        # sort result
        best = best.sort_values([team_col, player_col]).reset_index(drop=True) 
        return best

    def find_most_disciplined(self):
        """
        Task 9: Find the player(s) with the lowest number of yellow cards and red cards after playing for 90 minutes. The total minutes played should be more than 1000 minutes. Return player name, team, yellow cards, red cards, discipline score (cards per 90), position.


        note: The assignment PDF says finding the player with the HIGHEST discipline score.
        we follow the PDF requirement (highest discipline score).
        """
        df = self.cleaned_data.copy()

        player_col = "Player"
        team_col = "Squad"
        y_col = "Performance CrdY"
        r_col = "Performance CrdR"
        min_col = "Playing Time Min"
        nineties_col = "Playing Time 90s"
        pos_col = "Primary_Pos"
        
        # convert to num
        for c in [y_col, r_col, min_col, nineties_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        
        #filter players with >1000 min
        eligible = df[df[min_col]>1000].copy()

        # if 90s -> missing, zero
        if nineties_col not in eligible.columns:
            eligible[nineties_col] = eligible[min_col] / 90.0
        else:
            # if 90s is 0, compute from minutes
            eligible[nineties_col] = np.where(
                eligible[nineties_col] > 0,
                eligible[nineties_col],
                eligible[min_col] / 90.0
            )
        eligible["Discipline_Score"] = np.where(
            eligible[nineties_col] > 0,
            (eligible[r_col] + eligible[y_col] * 0.5) / eligible[nineties_col],
            0
        )
        # discipline score
        max_score = eligible["Discipline_Score"].max()

        result = eligible.loc[eligible["Discipline_Score"] == max_score, [player_col, team_col, y_col, r_col, "Discipline_Score", pos_col]].copy()
        result = result.sort_values([team_col, player_col]).reset_index(drop=True)
        return result

    def compare_leagues_attack(self):
        """
        Task 10: Compare the average number of goals scored by each league. Return a bar chart visualization and the average goals per 90 for each league.
        """
        df = self.cleaned_data.copy()

        goals_col = "Performance Gls"
        minutes_col = "Playing Time Min"
        league_col = "League"

        # make them to num
        df[goals_col] = pd.to_numeric(df[goals_col], errors="coerce").fillna(0)
        df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0)

        # Remove players who didnt play
        df = df[df[minutes_col] > 0]

        # groupby leauge with goal and minute
        league_stats = (
            df.groupby(league_col)
            .agg(
                total_goals=(goals_col, "sum"),
                total_minutes=(minutes_col, "sum")
            )
        )

        # average goals per 90
        league_stats["Avg_Goals_per_90"] = (
                league_stats["total_goals"] / league_stats["total_minutes"] * 90
        )
        #sort leagues
        league_stats = league_stats.sort_values("Avg_Goals_per_90", ascending=False)

        # bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(
            league_stats.index,
            league_stats["Avg_Goals_per_90"],
            color=[self.colors.get(lg) for lg in league_stats.index]
        )

        plt.title("Average Goals per 90 Minutes by League")
        plt.xlabel("League")
        plt.ylabel("Goals per 90")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        # Return dictionary
        league_dict = league_stats["Avg_Goals_per_90"].to_dict()

        return league_dict

    def compare_position_productivity(self):
        """
        Task 11: Compare the average number of goals scored and assists by each position.
        Return a plot with two subplots and the data dictionary for the plots:
          (1) Average G+A by Primary Position
          (2) Average G+A by Primary Position per League
        """
        df = self.cleaned_data.copy()

        pos_col = "Primary_Pos"
        league_col = "League"
        goals_col = "Performance Gls"
        ast_col = "Performance Ast"
        min_col = "Playing Time Min"

        # num check
        for c in [goals_col, ast_col, min_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # G+A per 90
        df["GA_per90"] = np.where(
            df[min_col] > 0,
            (df[goals_col] + df[ast_col]) / df[min_col] * 90,
            0
        )

        # remove players who didnt play
        df = df[df[min_col] > 0]

        # order positions
        pos_order = ["GK", "DF", "MF", "FW"]

        #  subplot 1
        avg_pos = (
            df.groupby(pos_col)["GA_per90"]
            .mean()
            .reindex(pos_order)
        )

        # subplot 2
        league_pos = (
            df.groupby([league_col, pos_col])["GA_per90"]
            .mean()
            .reset_index()
        )

        pivot = league_pos.pivot(index=pos_col, columns=league_col, values="GA_per90")
        pivot = pivot.reindex(pos_order)

        # plotting
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # first plot - overall position
        ax[0].bar(avg_pos.index, avg_pos.values)
        ax[0].set_title("Average G+A per 90 by Position")
        ax[0].set_xlabel("Position")
        ax[0].set_ylabel("G+A per 90")

        # second plot (position per league
        x = np.arange(len(pos_order))
        width = 0.15

        for i, league in enumerate(pivot.columns):
            ax[1].bar(
                x + i * width,
                pivot[league].fillna(0),
                width,
                label=league,
                color=self.colors.get(league)
            )

        ax[1].set_xticks(x + width)
        ax[1].set_xticklabels(pos_order)
        ax[1].set_title("Average G+A per 90 by Position per League")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        data = {
            "position_average": avg_pos.to_dict(),
            "league_position_average": pivot.to_dict()
        }

        return fig, data

    def age_curve_analysis(self):
        """
        Task 12: Compare the player age vs Goal Contribution Rate. Return a scatter plot visualization with a trend line and data.
        """
        df = self.cleaned_data.copy()

        age_col = "Age"
        rate_col = "Goal_Contribution_Rate"
        min_col = "Playing Time Min"

        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0)

        # remove bad rows
        df = df.dropna(subset=[age_col])
        if min_col in df.columns:
            df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0)
            df = df[df[min_col] > 0]

        x = df[age_col].values

        # make it per 90 so playing time difference is fair
        y = df[rate_col].values

        # correlation
        corr = float(pd.Series(x).corr(pd.Series(y)))

        # trendline
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.5)
        ax.plot(x_line, y_line, linestyle="--")

        ax.set_title("Age vs Goal Contribution Rate")
        ax.set_xlabel("Age")
        ax.set_ylabel("Goal Contribution Rate (G+A per 90)")

        plt.tight_layout()
        plt.show()

        data = {
            "correlation_age_vs_goal_contribution_rate": corr,
            "trendline_slope": float(m),
            "trendline_intercept": float(b)
        }

        return fig, data

    def league_defensive_discipline(self):
        """
        Task 13: Compare the average number of yellow cards and red cards by each league for defense. Return a plot with two subplots and the data dictionary for the plots.
        """
        # Replace with your code
        pass

    def youth_vs_veteran(self):
        """
        Task 14:
        Compare youth (<=21) vs veteran (>=32) performance across leagues.
        One subplot per position group (excluding Goalkeepers).
        """
        # Replace with your code
        pass

    def data_quality_report(self):
        """
        Task 15: Generate a report on data quality.
        """
        # Replace with your code
        pass

    def handle_transfers(self):
        """
        Task 16: Identify players with multiple entries and aggregate. Strategy: Sum counting stats, weighted average for rate stats.
        """
        # Replace with your code
        pass

    def __drawRadarChart(self, ax, player_data, player_name, color, attributes, all_players_data=None):
        """
        Helper method: Draws a single radar chart polygon on given axes.
        
        Input:
            ax: Matplotlib axes object (polar projection)
            player_data: Dict or Series of attribute values
            player_name: String for legend label
            color: Color string for the polygon
            attributes: List of attribute names (columns) to plot
            all_players_data: DataFrame of all players for normalization context
        """
        # Replace with your code
        pass

    def compare_players_radar(self, player1_name, player2_name, attributes, 
                             normalization='minmax', figsize=(10, 10), save_path=None):
        """
        Task 17: Compare two players using a radar chart across specified attributes.
        
        Input:
            player1_name: String, exact name of first player (e.g., "Erling Haaland")
            player2_name: String, exact name of second player (e.g., "Kylian Mbappé")
            attributes: List of column names to compare (e.g., ['Gls', 'Ast', 'xG', 'xAG', 'Shots'])
            normalization: 'minmax' (0-100 scale), 'raw' (actual values), or 'percentile' (league percentile)
            figsize: Tuple for figure size
            save_path: Optional path to save figure
            
        Output: Displays radar chart (and saves if path provided). Returns comparison DataFrame.
        """
        # Replace with your code
        pass

    def _aggregate_player_stats(self, player_df):
        """Helper to sum stats for players with multiple rows (transfers)"""
        # Replace with your code
        pass
    
if __name__ == "__main__":

    analysis = EuropeanFootballAnalysis()
    analysis.scrape()  # sets raw_data
    analysis.clean_data()   # returns + stores self.clean_data
    analysis.add_derived_metrics()
    top_scorers = analysis.find_top_scorer()
    # print(top_scorers)
    # print(analysis.find_playmaker())
    #print(analysis.find_efficient_striker())


    # print(analysis.cleaned_data[["Player", "Performance Gls","Performance Ast","Playing Time Min","Playing Time MP","Minutes_per_Game","Goal_Contribution_Rate"]].head(10))    
    #print(analysis.cleaned_data.head(20))
    #print(analysis.cleaned_data[["Player","Nation","Comp","League","Playing Time Min"]].head(20))
    #print(analysis.cleaned_data[["Player", "Pos", "Primary_Pos","Primary_Pos_Full"]].head(20))
    
    #analysis.add_derived_metrics()
    # fig, d = analysis.compare_position_productivity()
    # print(d["position_average"])
    # print(list(d["league_position_average"].keys()))
    # fig12, d12 = analysis.age_curve_analysis()
    # print(d12)
    analysis.add_derived_metrics()
    print(analysis.cleaned_data[["Playing Time Min","Playing Time MP","Playing Time 90s","Minutes_per_Game","Goal_Contribution_Rate"]].head())