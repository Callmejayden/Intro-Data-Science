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

        # Remove special characters (keeps letters, numbers, underscores and whitespace)
        self.cleaned_data = self.cleaned_data.replace(to_replace=r'[^\w\s]', value='', regex=True)


        # Strip whitespace from common string columns
        cols_string = ["Squad", "League", "Nation", "Pos", "Player", "Team"]
        for c in cols_string:
            if c in self.cleaned_data.columns and self.cleaned_data[c].dtype == object:
                self.cleaned_data[c] = self.cleaned_data[c].str.strip()


        # Convert numeric columns with commas to numeric data
        cols_numeric = ["Age", "Gls", "Ast", "Min", "Matches", "Yellow Cards", "Red Cards", "xG", "xAG", "Shots", "xG per Shot", "xAG per Shot", "xG per 90", "xAG per 90", "G+A per 90", "Yellow Cards per 90", "Red Cards per 90", "Matches per 90"]
        for c in cols_numeric:
            if c in self.cleaned_data.columns:
                if self.cleaned_data[c].dtype == object:
                    # Replace commas and strip whitespace, then convert to numeric
                    self.cleaned_data[c] = self.cleaned_data[c].str.replace(',', '', regex=False).str.strip()
                self.cleaned_data[c] = pd.to_numeric(self.cleaned_data[c], errors='coerce')


        # Populate primary position column using helper
        if "Pos" in self.cleaned_data.columns:
            self.cleaned_data["Primary_Pos"] = self.cleaned_data["Pos"].apply(self._extract_primary_position)
        else:
            self.cleaned_data["Primary_Pos"] = np.nan

        # Additional parsing (nation, transfers, etc.) can be added here

        # Map primary position to full name
        self.cleaned_data["Primary_Pos_Full"] = self.cleaned_data["Primary_Pos"].map(self.position_map)

        # Map team to color
        self.cleaned_data["Team_Color"] = self.cleaned_data["Squad"].map(self.team_colors)
        
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
        # Replace with your code
        pass

    def find_top_scorer(self):
        """
        Task 5: Find the player(s) with the highest number of goals. Return player name, team, goals, league.
        """
        # Replace with your code
        pass

    def find_playmaker(self):
        """
        Task 6: Find the player(s) with the highest number of assists. Return player name, team, assists, league.
        """
        # Replace with your code
        pass

    def find_ironman(self):
        """
        Task 7: Find the player(s) with the highest number of minutes played. Return player name, team, minutes played, matches played.
        """
        # Replace with your code
        pass

    def find_efficient_striker(self):
        """
        Task 8: Find the forward(s) who is most efficent (most goals per minutes played) and has played more than 1000 minutes. Return player name, team, goals, minutes played, goals per minute.
        """
        # Replace with your code
        pass


    def find_most_disciplined(self):
        """
        Task 9: Find the player(s) with the lowest number of yellow cards and red cards after playing for 90 minutes. The total minutes played should be more than 1000 minutes. Return player name, team, yellow cards, red cards, discipline score (cards per 90), position.
        """
        # Replace with your code
        pass

    def compare_leagues_attack(self):
        """
        Task 10: Compare the average number of goals scored by each league. Return a bar chart visualization and the average goals per 90 for each league.
        """
        # Replace with your code
        pass

    def compare_position_productivity(self):
        """
        Task 11: Compare the average number of goals scored and assists by each position.
        Return a plot with two subplots and the data dictionary for the plots:
          (1) Average G+A by Primary Position
          (2) Average G+A by Primary Position per League
        """
        # Replace with your code
        pass

    def age_curve_analysis(self):
        """
        Task 12: Compare the player age vs Goal Contribution Rate. Return a scatter plot visualization with a trend line and data.
        """
        # Replace with your code
        pass

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
    #print(analysis.raw_data.head())


    analysis.clean_data()   # returns + stores self.clean_data
    print(analysis.cleaned_data.head(20))

    #analysis._extract_primary_position("Pos")  # adds Primary_Pos column to cleaned_data

    #print(analysis.cleaned_data[["Player", "Pos", "Primary_Pos","Primary_Pos_Full"]].head(20))
    

    #analysis.add_derived_metrics()


