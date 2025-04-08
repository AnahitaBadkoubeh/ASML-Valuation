import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ASMLDCFAnalysis:
    def __init__(self):
        # Base assumptions
        self.base_assumptions = {
            'revenue_growth': 0.15,  # 15% annual growth
            'ebitda_margin': 0.35,   # 35% EBITDA margin
            'tax_rate': 0.15,        # 15% tax rate
            'capex_percent': 0.10,   # 10% of revenue
            'working_capital_percent': 0.15,  # 15% of revenue
            'wacc': 0.10,            # 10% WACC
            'terminal_growth': 0.03  # 3% terminal growth
        }
        
        # Bull case assumptions
        self.bull_assumptions = {
            'revenue_growth': 0.20,  # 20% annual growth
            'ebitda_margin': 0.40,   # 40% EBITDA margin
            'tax_rate': 0.15,
            'capex_percent': 0.08,   # 8% of revenue
            'working_capital_percent': 0.12,  # 12% of revenue
            'wacc': 0.09,            # 9% WACC
            'terminal_growth': 0.04  # 4% terminal growth
        }
        
        # Bear case assumptions
        self.bear_assumptions = {
            'revenue_growth': 0.10,  # 10% annual growth
            'ebitda_margin': 0.30,   # 30% EBITDA margin
            'tax_rate': 0.15,
            'capex_percent': 0.12,   # 12% of revenue
            'working_capital_percent': 0.18,  # 18% of revenue
            'wacc': 0.11,            # 11% WACC
            'terminal_growth': 0.02  # 2% terminal growth
        }
        
        # Default values in case yfinance fails
        self.current_price = 800.0  # Default price in euros
        self.current_revenue = 20_000_000_000  # Default revenue in euros
        
        try:
            # Get current market data
            self.asml = yf.Ticker("ASML")
            self.current_price = self.asml.info.get('currentPrice', self.current_price)
            self.current_revenue = self.asml.info.get('totalRevenue', self.current_revenue)
        except Exception as e:
            print(f"Warning: Could not fetch data from yfinance. Using default values. Error: {str(e)}")
        
        # Comparable companies
        self.comparables = ['LRCX', 'AMAT', 'KLAC', 'TER']
        
    def calculate_fcf(self, assumptions, years=5):
        """Calculate free cash flow projections"""
        revenue = self.current_revenue
        fcf_projections = []
        
        for year in range(1, years + 1):
            # Project revenue
            revenue *= (1 + assumptions['revenue_growth'])
            
            # Calculate EBITDA
            ebitda = revenue * assumptions['ebitda_margin']
            
            # Calculate taxes
            taxes = ebitda * assumptions['tax_rate']
            
            # Calculate capex
            capex = revenue * assumptions['capex_percent']
            
            # Calculate working capital changes
            working_capital = revenue * assumptions['working_capital_percent']
            
            # Calculate FCF
            fcf = ebitda - taxes - capex - working_capital
            fcf_projections.append(fcf)
            
        return fcf_projections
    
    def calculate_terminal_value(self, last_fcf, assumptions):
        """Calculate terminal value using Gordon Growth Model"""
        return (last_fcf * (1 + assumptions['terminal_growth'])) / \
               (assumptions['wacc'] - assumptions['terminal_growth'])
    
    def calculate_present_value(self, fcf_projections, terminal_value, wacc):
        """Calculate present value of cash flows"""
        pv_cash_flows = []
        for i, fcf in enumerate(fcf_projections):
            pv = fcf / ((1 + wacc) ** (i + 1))
            pv_cash_flows.append(pv)
        
        pv_terminal = terminal_value / ((1 + wacc) ** len(fcf_projections))
        return sum(pv_cash_flows) + pv_terminal
    
    def run_dcf_analysis(self):
        """Run DCF analysis for all scenarios"""
        results = {}
        
        for scenario, assumptions in [
            ('Base', self.base_assumptions),
            ('Bull', self.bull_assumptions),
            ('Bear', self.bear_assumptions)
        ]:
            fcf_projections = self.calculate_fcf(assumptions)
            terminal_value = self.calculate_terminal_value(fcf_projections[-1], assumptions)
            enterprise_value = self.calculate_present_value(fcf_projections, terminal_value, assumptions['wacc'])
            
            results[scenario] = {
                'Enterprise Value': enterprise_value,
                'FCF Projections': fcf_projections,
                'Terminal Value': terminal_value
            }
        
        return results
    
    def comparable_analysis(self):
        """Perform comparable company analysis"""
        comp_data = []
        
        # Default values for comparable analysis
        default_metrics = {
            'EV/EBITDA': 20.0,
            'P/E': 30.0,
            'P/S': 10.0,
            'ROE': 0.25
        }
        
        for ticker in self.comparables:
            try:
                company = yf.Ticker(ticker)
                info = company.info
                
                comp_data.append({
                    'Ticker': ticker,
                    'EV/EBITDA': info.get('enterpriseToEbitda', default_metrics['EV/EBITDA']),
                    'P/E': info.get('trailingPE', default_metrics['P/E']),
                    'P/S': info.get('priceToSalesTrailing12Months', default_metrics['P/S']),
                    'ROE': info.get('returnOnEquity', default_metrics['ROE'])
                })
            except Exception as e:
                print(f"Warning: Could not fetch data for {ticker}. Using default values. Error: {str(e)}")
                comp_data.append({
                    'Ticker': ticker,
                    **default_metrics
                })
        
        return pd.DataFrame(comp_data)
    
    def plot_results(self, results):
        """Plot DCF results and comparable analysis"""
        # Create a figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: FCF projections
        ax1 = plt.subplot(2, 2, 1)
        for scenario, data in results.items():
            ax1.plot(data['FCF Projections'], label=f'{scenario} Case')
        ax1.set_title('ASML Free Cash Flow Projections')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Free Cash Flow (€ billions)')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['2025', '2026', '2027', '2028', '2029'])
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Enterprise Value
        ax2 = plt.subplot(2, 2, 2)
        scenarios = list(results.keys())
        values = [data['Enterprise Value'] / 1e9 for data in results.values()]
        ax2.bar(scenarios, values)
        ax2.set_title('ASML Enterprise Value by Scenario')
        ax2.set_ylabel('Enterprise Value (€ billions)')
        ax2.grid(True)
        
        # Plot 3: Comparable Company Analysis
        ax3 = plt.subplot(2, 2, (3, 4))
        comp_data = self.comparable_analysis()
        
        # Plot multiple metrics for comparable companies
        metrics = ['EV/EBITDA', 'P/E', 'P/S', 'ROE']
        x = np.arange(len(comp_data['Ticker']))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax3.bar(x + i*width, comp_data[metric], width, label=metric)
        
        ax3.set_title('Comparable Company Analysis')
        ax3.set_xlabel('Companies')
        ax3.set_ylabel('Valuation Metrics')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(comp_data['Ticker'])
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    analysis = ASMLDCFAnalysis()
    
    # Run DCF analysis
    dcf_results = analysis.run_dcf_analysis()
    
    # Print DCF results
    print("\nDCF Analysis Results:")
    for scenario, data in dcf_results.items():
        print(f"\n{scenario} Case:")
        print(f"Enterprise Value: €{data['Enterprise Value']:,.2f}")
        print(f"Terminal Value: €{data['Terminal Value']:,.2f}")
        print("FCF Projections:")
        for year, fcf in enumerate(data['FCF Projections'], 1):
            print(f"Year {year}: €{fcf:,.2f}")
    
    # Run comparable analysis
    comp_analysis = analysis.comparable_analysis()
    print("\nComparable Company Analysis:")
    print(comp_analysis)
    
    # Plot results
    analysis.plot_results(dcf_results)

if __name__ == "__main__":
    main() 