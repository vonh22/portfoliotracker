import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from requests.exceptions import Timeout
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache


# Configure Streamlit page settings
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# Initialize session state for transactions and goals
if 'transactions' not in st.session_state:
    st.session_state['transactions'] = []
if 'goals' not in st.session_state:
    st.session_state['goals'] = []

# File paths for data storage
TRANSACTIONS_FILE = 'transactions.json'
GOALS_FILE = 'goals.json'

def load_data():
    """Load saved data from JSON files"""
    try:
        if os.path.exists(TRANSACTIONS_FILE):
            with open(TRANSACTIONS_FILE, 'r') as f:
                st.session_state['transactions'] = json.load(f)
                st.write(f"Loaded {len(st.session_state['transactions'])} transactions")
        
        if os.path.exists(GOALS_FILE):
            with open(GOALS_FILE, 'r') as f:
                st.session_state['goals'] = json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def save_data():
    """Save data to JSON files"""
    try:
        with open(TRANSACTIONS_FILE, 'w') as f:
            json.dump(st.session_state['transactions'], f)
        
        with open(GOALS_FILE, 'w') as f:
            json.dump(st.session_state['goals'], f)
        
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def get_stock_price(symbol):
    """Fetch current stock price using yfinance with multiple retry attempts"""
    try:
        # Create ticker object
        stock = yf.Ticker(symbol)
        
        # First try to get regular market price
        current_price = stock.info.get('regularMarketPrice')
        if current_price is not None and current_price > 0:
            return current_price
            
        # If that fails, try to get the last closing price from history
        hist = stock.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
            
        # If both methods fail, try to get the previous closing price
        prev_close = stock.info.get('previousClose')
        if prev_close is not None and prev_close > 0:
            st.warning(f"Using previous closing price for {symbol}")
            return prev_close
            
        st.warning(f"Could not fetch current price for {symbol}. Please check the symbol is correct.")
        return None
        
    except Exception as e:
        st.error(f"Error fetching price for {symbol}. Error: {str(e)}")
        return None
    

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Cache for storing analysis results
analysis_cache = {}
CACHE_EXPIRY = timedelta(hours=24)  # Cache results for 24 hours

def get_portfolio_hash(portfolio_data):
    """
    Create a hash of portfolio data to use as cache key
    """
    # Sort and convert to string to ensure consistent hashing
    portfolio_str = json.dumps(sorted(portfolio_data, key=lambda x: x['Symbol']))
    return hash(portfolio_str)

@lru_cache(maxsize=100)
def get_holdings_summary(portfolio_tuple):
    """
    Calculate holdings summary with caching
    Portfolio data passed as tuple for hashability
    """
    portfolio_data = json.loads(portfolio_tuple)
    total_value = sum(float(holding['Current Value']) for holding in portfolio_data)
    
    holdings_summary = []
    for holding in portfolio_data:
        weight = (float(holding['Current Value']) / total_value) * 100
        holdings_summary.append(
            f"- {holding['Symbol']}: {holding['Quantity']} shares at ${holding['Current Price']}, "
            f"Portfolio Weight: {weight:.2f}%"
        )
    
    return "\n".join(holdings_summary)

def get_optimized_prompt(holdings_summary, analysis_type):
    """
    Generate optimized, token-efficient prompts
    """
    base_prompts = {
        "general": f"""Analyze portfolio holdings and weights:
{holdings_summary}

Provide concise analysis of:
1. Diversification/concentration
2. Risk assessment
3. Rebalancing recommendations
4. Position sizing""",
        
        "risk": f"""Portfolio holdings:
{holdings_summary}

Analyze:
1. Position-size risks
2. Sector concentration
3. Market exposure
4. Risk management steps
5. Overweight positions""",
        
        "opportunities": f"""Current portfolio:
{holdings_summary}

Analyze:
1. Weight distribution
2. Expansion opportunities
3. Underrepresented sectors
4. Balance improvement ideas"""
    }
    
    return base_prompts[analysis_type]

def get_ai_analysis(portfolio_data, analysis_type="general"):
    """
    Get AI analysis of the portfolio using OpenAI's GPT-4 with optimization
    """
    try:
        # Convert portfolio data to hashable format
        portfolio_tuple = json.dumps(portfolio_data, sort_keys=True)
        
        # Generate cache key
        cache_key = (get_portfolio_hash(portfolio_data), analysis_type)
        
        # Check cache
        if cache_key in analysis_cache:
            cached_result, timestamp = analysis_cache[cache_key]
            if datetime.now() - timestamp < CACHE_EXPIRY:
                return cached_result
        
        # Get cached or calculate holdings summary
        holdings_summary = get_holdings_summary(portfolio_tuple)
        
        # Get optimized prompt
        prompt = get_optimized_prompt(holdings_summary, analysis_type)
        
        # Make API call with optimized parameters
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Financial advisor providing concise portfolio analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Lower temperature for more consistent responses
            max_tokens=500,   # Limit response length
            presence_penalty=-0.5,  # Encourage focused responses
            frequency_penalty=0.3   # Slight penalty for repetition
        )
        
        result = response.choices[0].message.content
        
        # Cache the result
        analysis_cache[cache_key] = (result, datetime.now())
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

def clear_cache():
    """
    Clear expired cache entries
    """
    current_time = datetime.now()
    expired_keys = [
        key for key in analysis_cache 
        if current_time - analysis_cache[key][1] >= CACHE_EXPIRY
    ]
    for key in expired_keys:
        del analysis_cache[key]


def calculate_portfolio_value():
    """Calculate current portfolio value and holdings"""
    holdings = {}
    
    # First pass: calculate net quantities
    for transaction in st.session_state['transactions']:
        symbol = transaction['symbol']
        quantity = float(transaction['quantity'])
        
        if transaction['type'] == 'buy':
            holdings[symbol] = holdings.get(symbol, 0) + quantity
        elif transaction['type'] == 'sell':
            holdings[symbol] = holdings.get(symbol, 0) - quantity

    # Second pass: calculate current values
    total_value = 0
    current_holdings = []
    
    for symbol, quantity in holdings.items():
        if quantity > 0:  # Only include positive holdings
            current_price = get_stock_price(symbol)
            if current_price:
                value = current_price * quantity
                total_value += value
                current_holdings.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Current Price': current_price,
                    'Current Value': value
                })
    
    return total_value, pd.DataFrame(current_holdings)

def calculate_performance():
    """Calculate performance metrics for each holding"""
    holdings = {}
    
    # Calculate average purchase price and total quantity for each symbol
    for transaction in st.session_state['transactions']:
        symbol = transaction['symbol']
        quantity = float(transaction['quantity'])
        price = float(transaction['price'])
        
        if transaction['type'] == 'buy':
            if symbol not in holdings:
                holdings[symbol] = {'total_cost': 0, 'total_quantity': 0}
            holdings[symbol]['total_cost'] += price * quantity
            holdings[symbol]['total_quantity'] += quantity
        elif transaction['type'] == 'sell':
            if symbol in holdings:
                holdings[symbol]['total_quantity'] -= quantity

    # Calculate performance metrics
    performance_data = []
    for symbol, data in holdings.items():
        if data['total_quantity'] > 0:  # Only include current holdings
            current_price = get_stock_price(symbol)
            if current_price:
                avg_price = data['total_cost'] / data['total_quantity']
                gain_loss_pct = ((current_price - avg_price) / avg_price) * 100
                gain_loss_value = (current_price - avg_price) * data['total_quantity']
                
                performance_data.append({
                    'Symbol': symbol,
                    'Average Cost': avg_price,
                    'Current Price': current_price,
                    'Quantity': data['total_quantity'],  # Added this line
                    'Gain/Loss (%)': gain_loss_pct,
                    'Gain/Loss ($)': gain_loss_value,
                    'Current Value': current_price * data['total_quantity']
                })
    
    return pd.DataFrame(performance_data)

def validate_transaction(symbol, quantity, price):
    """Validate transaction inputs"""
    if not symbol:
        st.error("Please enter a valid symbol")
        return False
    if quantity <= 0:
        st.error("Quantity must be greater than 0")
        return False
    if price <= 0:
        st.error("Price must be greater than 0")
        return False
    return True

def main():
    st.title("Portfolio Tracker")
    
    # Load saved data
    load_data()
    
    # Sidebar menu
    menu = st.sidebar.selectbox(
        "Menu",
        ["Portfolio Overview", "Add Transaction", "Goals", "Performance", "AI Analysis"]
    )
    
    if menu == "Portfolio Overview":
        # Custom CSS for styling
        st.markdown("""
            <style>
            .metric-card {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                text-align: center;
                backdrop-filter: blur(10px);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #00ff9d;
            }
            .metric-label {
                font-size: 14px;
                opacity: 0.85;
            }
            .holdings-card {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                margin-top: 20px;
                backdrop-filter: blur(10px);
            }
            div[data-testid="stDataFrame"] div[role="cell"] {
                color: inherit !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Page header with subtle divider
        st.markdown("# 📊 Portfolio Overview")
        st.markdown("---")

        # Calculate portfolio metrics
        total_value, holdings_df = calculate_portfolio_value()
        
        # Top metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">💰 ${:,.2f}</div>
                    <div class="metric-label">Total Portfolio Value</div>
                </div>
            """.format(total_value), unsafe_allow_html=True)
            
        # Calculate daily change if holdings exist
        if not holdings_df.empty:
            daily_change = holdings_df['Current Value'].sum() * 0.01  # Example calculation
            daily_change_pct = (daily_change / total_value) * 100
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">📈 ${abs(daily_change):,.2f}</div>
                        <div class="metric-label">Today's Change ({daily_change_pct:.2f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                num_holdings = len(holdings_df)
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">🏢 {num_holdings}</div>
                        <div class="metric-label">Total Holdings</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Holdings section
        if not holdings_df.empty:
            st.markdown("""
                <div class="holdings-card">
                    <h3>Current Holdings</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Format the dataframe
            display_df = holdings_df.copy()
            display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
            display_df['Current Value'] = display_df['Current Value'].map('${:,.2f}'.format)
            display_df = display_df.sort_values('Current Value', ascending=False)
            
            # Display holdings in a clean table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Symbol": st.column_config.TextColumn(
                        "Symbol",
                        help="Stock ticker symbol",
                        width="medium"
                    ),
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity",
                        help="Number of shares",
                        format="%.2f"
                    ),
                    "Current Price": st.column_config.TextColumn(
                        "Current Price",
                        help="Latest market price",
                        width="medium"
                    ),
                    "Current Value": st.column_config.TextColumn(
                        "Current Value",
                        help="Total value of holding",
                        width="medium"
                    )
                }
            )
            
            # Asset allocation chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="holdings-card">
                    <h3>Asset Allocation</h3>
                </div>
            """, unsafe_allow_html=True)
            
            fig = px.pie(
                holdings_df,
                values='Current Value',
                names='Symbol',
                title='',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👋 Welcome! Add some transactions to start tracking your portfolio.")
            
    elif menu == "Add Transaction":
        st.header("Add Transaction")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                date = st.date_input("Date", datetime.today())
                symbol = st.text_input("Symbol").upper()
                transaction_type = st.selectbox("Type", ["buy", "sell", "dividend"])
            
            with col2:
                quantity = st.number_input("Quantity", min_value=0.0, step=0.01)
                price = st.number_input("Price per share", min_value=0.0, step=0.01)
                fees = st.number_input("Fees", min_value=0.0, step=0.01)
            
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                if validate_transaction(symbol, quantity, price):
                    # Create new transaction
                    new_transaction = {
                        'date': date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'type': transaction_type,
                        'quantity': quantity,
                        'price': price,
                        'fees': fees
                    }
                    
                    # Add to session state
                    st.session_state['transactions'].append(new_transaction)
                    
                    # Save to file
                    if save_data():
                        st.success(f"Transaction added successfully! Total transactions: {len(st.session_state['transactions'])}")
                        # Display current transactions
                        st.subheader("Recent Transactions")
                        transactions_df = pd.DataFrame(st.session_state['transactions'][-5:])  # Show last 5 transactions
                        st.dataframe(transactions_df)
                    else:
                        st.error("Failed to save transaction. Please try again.")
        
    elif menu == "Goals":
        st.markdown("# 🎯 Financial Goals")
        st.markdown("---")
        
        with st.form("goal_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                goal_name = st.text_input("Goal Name")
                target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=1000.0)
            
            with col2:
                target_date = st.date_input("Target Date", datetime.today() + timedelta(days=365))
                goal_description = st.text_area("Description (Optional)", height=100)
            
            if st.form_submit_button("Add Goal"):
                if goal_name and target_amount > 0:
                    goal = {
                        'name': goal_name,
                        'target_amount': target_amount,
                        'target_date': target_date.strftime('%Y-%m-%d'),
                        'description': goal_description
                    }
                    st.session_state['goals'].append(goal)
                    save_data()
                    st.success("Goal added successfully!")
                else:
                    st.error("Please enter a valid goal name and target amount.")
        
        if st.session_state['goals']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="holdings-card">
                    <h3>Goal Progress</h3>
                </div>
            """, unsafe_allow_html=True)
            
            total_value, _ = calculate_portfolio_value()
            
            # Create a list to store goals to remove
            goals_to_remove = []
            
            for idx, goal in enumerate(st.session_state['goals']):
                progress_ratio = min(total_value / goal['target_amount'], 1.0)
                progress_pct = progress_ratio * 100
                
                days_remaining = (datetime.strptime(goal['target_date'], '%Y-%m-%d') - datetime.now()).days
                
                col1, col2, col3 = st.columns([1.5, 1, 0.5])
                with col1:
                    st.markdown(f"### {goal['name']}")
                    if 'description' in goal and goal['description']:
                        st.markdown(f'<div style="color: #666; font-style: italic; margin-bottom: 10px;">{goal["description"]}</div>', unsafe_allow_html=True)
                    st.progress(progress_ratio)
                    st.markdown(f"""
                        **Progress:** ${total_value:,.2f} / ${goal['target_amount']:,.2f} ({progress_pct:.1f}%)
                        \n**Days Remaining:** {max(days_remaining, 0)} days
                    """)
                
                with col2:
                    # Calculate monthly targets
                    if days_remaining > 0:
                        remaining_amount = goal['target_amount'] - total_value
                        months_remaining = days_remaining / 30.44
                        monthly_target = remaining_amount / months_remaining
                        
                        if remaining_amount > 0:
                            metric_label = "Monthly Investment Needed"
                            metric_value = f"${max(monthly_target, 0):,.2f}"
                        else:
                            metric_label = "Goal Achieved!"
                            metric_value = "🎉"
                        
                        st.markdown("""
                            <div class="metric-card">
                                <div class="metric-label">{}</div>
                                <div class="metric-value">{}</div>
                            </div>
                        """.format(metric_label, metric_value), unsafe_allow_html=True)
                
                with col3:
                    # Add remove button for each goal
                    if st.button("🗑️", key=f"delete_goal_{idx}"):
                        goals_to_remove.append(idx)
                
                st.markdown("---")
            
            # Remove marked goals and save changes
            if goals_to_remove:
                for idx in sorted(goals_to_remove, reverse=True):
                    removed_goal = st.session_state['goals'].pop(idx)
                    st.success(f"Removed goal: {removed_goal['name']}")
                save_data()
                st.rerun()  # Rerun the app to update the display
        else:
            st.info("👋 Add your first financial goal to start tracking your progress!")
                
    elif menu == "Performance":
        # Page header
        st.markdown("# 📈 Performance Analysis")
        st.markdown("---")
        
        performance_df = calculate_performance()
        
        if not performance_df.empty:
            # Calculate overall portfolio metrics
            total_invested = (performance_df['Average Cost'] * performance_df['Quantity']).sum()
            total_current = performance_df['Current Value'].sum()
            total_gain_loss = total_current - total_invested
            total_gain_loss_pct = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0
            best_performer = performance_df.loc[performance_df['Gain/Loss (%)'].idxmax()]
            worst_performer = performance_df.loc[performance_df['Gain/Loss (%)'].idxmin()]
            
            # Top metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Total Gain/Loss</div>
                    </div>
                """.format(
                    f"{'↑' if total_gain_loss >= 0 else '↓'} ${abs(total_gain_loss):,.2f}"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Return (%)</div>
                    </div>
                """.format(
                    f"{'↑' if total_gain_loss_pct >= 0 else '↓'} {abs(total_gain_loss_pct):.2f}%"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">${:,.2f}</div>
                        <div class="metric-label">Total Invested</div>
                    </div>
                """.format(total_invested), unsafe_allow_html=True)

            # Best/Worst Performers
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Best Performer</div>
                        <div class="metric-value">{} (↑{:.2f}%)</div>
                        <div class="metric-label">Gain: ${:,.2f}</div>
                    </div>
                """.format(
                    best_performer['Symbol'],
                    best_performer['Gain/Loss (%)'],
                    best_performer['Gain/Loss ($)']
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Worst Performer</div>
                        <div class="metric-value">{} (↓{:.2f}%)</div>
                        <div class="metric-label">Loss: ${:,.2f}</div>
                    </div>
                """.format(
                    worst_performer['Symbol'],
                    abs(worst_performer['Gain/Loss (%)']),
                    abs(worst_performer['Gain/Loss ($)'])
                ), unsafe_allow_html=True)

            # Performance Charts
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Waterfall chart for individual asset performance
            performance_df_sorted = performance_df.sort_values('Gain/Loss (%)', ascending=True)
            fig_waterfall = px.bar(
                performance_df_sorted,
                x='Symbol',
                y='Gain/Loss (%)',
                title='Asset Performance Breakdown',
                labels={'Symbol': 'Asset', 'Gain/Loss (%)': 'Return (%)'},
                color='Gain/Loss (%)',
                color_continuous_scale=['#FF4B4B', '#ffffff', '#2ED477']
            )
            fig_waterfall.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Asset Performance Table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="holdings-card">
                    <h3>Detailed Performance Metrics</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Format the dataframe for display
            display_df = performance_df.copy()
            display_df['Average Cost'] = display_df['Average Cost'].map('${:,.2f}'.format)
            display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
            display_df['Gain/Loss (%)'] = display_df.apply(
                lambda x: f"{'↑' if x['Gain/Loss (%)'] >= 0 else '↓'} {abs(x['Gain/Loss (%)']):,.2f}%",
                axis=1
            )
            display_df['Gain/Loss ($)'] = display_df.apply(
                lambda x: f"{'↑' if x['Gain/Loss ($)'] >= 0 else '↓'} ${abs(x['Gain/Loss ($)']):,.2f}",
                axis=1
            )
            display_df['Current Value'] = display_df['Current Value'].map('${:,.2f}'.format)
            
            # Display the table with custom styling
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Symbol": st.column_config.TextColumn(
                        "Symbol",
                        help="Stock ticker symbol",
                        width="medium"
                    ),
                    "Average Cost": st.column_config.TextColumn(
                        "Avg. Cost",
                        help="Average purchase price per share",
                        width="medium"
                    ),
                    "Current Price": st.column_config.TextColumn(
                        "Current",
                        help="Current market price",
                        width="medium"
                    ),
                    "Gain/Loss (%)": st.column_config.TextColumn(
                        "Return",
                        help="Percentage return",
                        width="medium"
                    ),
                    "Gain/Loss ($)": st.column_config.TextColumn(
                        "P/L",
                        help="Profit/Loss in dollars",
                        width="medium"
                    ),
                    "Current Value": st.column_config.TextColumn(
                        "Value",
                        help="Current holding value",
                        width="medium"
                    )
                }
            )
            
            # Extra visualization: Value Distribution
            st.markdown("<br>", unsafe_allow_html=True)
            fig_treemap = px.treemap(
                performance_df,
                path=[px.Constant("Portfolio"), 'Symbol'],
                values='Current Value',
                color='Gain/Loss (%)',
                color_continuous_scale=['#FF4B4B', '#ffffff', '#2ED477'],
                title='Portfolio Value Distribution'
            )
            fig_treemap.update_layout(height=400)
            st.plotly_chart(fig_treemap, use_container_width=True)

    elif menu == "AI Analysis":
        st.markdown("# 🤖 AI Portfolio Analysis")
        st.markdown("---")
        
        # Get current portfolio data
        total_value, holdings_df = calculate_portfolio_value()
        
        if not holdings_df.empty:
            # Initialize session state for chat history
            if 'ai_messages' not in st.session_state:
                st.session_state.ai_messages = []
                
            # Analysis type selector
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["general", "risk", "opportunities"],
                format_func=lambda x: x.capitalize()
            )
            
            if st.button("Generate Analysis"):
                with st.spinner('Analyzing portfolio...'):
                    # Convert DataFrame to list of dicts for the AI function
                    portfolio_data = holdings_df.to_dict('records')
                    analysis = get_ai_analysis(portfolio_data, analysis_type)
                    
                    # Add to chat history
                    st.session_state.ai_messages.append({
                        "role": "assistant",
                        "content": analysis,
                        "type": analysis_type
                    })
            
            # Display chat history
            for message in st.session_state.ai_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    st.caption(f"Analysis Type: {message['type'].capitalize()}")
            
            # Clear chat history button
            if st.button("Clear Analysis History"):
                st.session_state.ai_messages = []
                st.rerun()
            
    else:
        st.info("👋 Add some holdings to your portfolio to get AI-powered analysis and recommendations!")   



if __name__ == "__main__":
    main()