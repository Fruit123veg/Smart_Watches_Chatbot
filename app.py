import streamlit as st
import pandas as pd
import os
import time
import datetime
import traceback
import matplotlib.pyplot as plt
import requests
import json
from openai import OpenAI
import holidays

# Page configurations for a modern, professional look
st.set_page_config(page_title="Smart_Watches_Chatbot", layout="wide")

DEBUG_MODE = True

# Securely reading tokens from environment context variables with NO hardcoded fallbacks
hf_token = os.environ.get("TITAN_API_KEY", "").strip()
redshift_api_key = os.environ.get("REDSHIFT_API_KEY", "").strip()

# --- SIDEBAR: SYSTEM LOGS & ACTIONS ---
with st.sidebar:
    st.markdown("### 🛠️ System Control & Logs")
    if not hf_token:  
        st.error("❌ 'TITAN_API_KEY' is missing!")  
    else:  
        st.info("🔑 Titan Token detected!")  
               
    if not redshift_api_key:  
        st.error("❌ 'REDSHIFT_API_KEY' is missing!")  
    else:  
        st.success("🔑 Redshift Token detected!")

# Connect via OpenAI structure pointing directly to Titan Enterprise gateway
client = OpenAI(
    base_url="https://ai.titan.in/gateway",
    api_key=hf_token
)

# Premium Heading styled in White
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🌐 Enterprise Intelligence Engine</h1>", unsafe_allow_html=True)
st.write("")

# Create the dedicated history folder if it doesn't exist
HISTORY_FOLDER = "chat_history"
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)

# Unified single history session list to keep items, queries, and descriptions perfectly aligned
if 'interaction_history' not in st.session_state:
    st.session_state['interaction_history'] = []

def run_redshift_query(sql_query, api_key):
    url = "https://yizdz9le4f.execute-api.ap-south-1.amazonaws.com/AI/AI_data_story"
    payload = json.dumps({"SQL": sql_query})
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }
    response = requests.request("POST", url, headers=headers, data=payload)
         
    if response.status_code == 200:  
        raw_json = json.loads(response.text)  
        try:  
            records = raw_json["Result"]["data"]  
        except (KeyError, TypeError):  
            records = raw_json.get("data", raw_json)  
                       
        df = pd.DataFrame(records)  
                     
        if df.empty:  
            return df  
                       
        # Clean up column names in case Redshift returns lowercased aggregation expressions  
        df.columns = [c.lower().split('.')[-1].replace('sum(', '').replace(')', '').strip() for c in df.columns]  
                       
        # Convert any numeric looking metrics safely to floats  
        numeric_cols = [  
            'service_revenue', 'others', 'wdc', 'no_discount',  
            'retail_revenue', 'str', 'retail_cust', 'growth_percent',  
            'contribution_pct', 'contribution_percent'  
        ]  
        for col in df.columns:  
            if any(n in col for n in numeric_cols) or 'percent' in col or 'growth' in col or 'contribution' in col:  
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)  
                     
        if 'date' in df.columns:  
            df['date'] = df['date'].astype(str).str.strip()  
            df['date'] = pd.to_datetime(df['date'], errors='coerce')  
                       
        return df  
    else:  
        raise Exception(f"Gateway Error {response.status_code}: {response.text}")
def get_dynamic_indian_festivals(year):
    """
    Dynamically fetches the exact month and details of shifting Indian festivals 
    for any given year using the 'holidays' library.
    """
    # Create the Indian Holiday entity (with subdivisions if needed, e.g., TN, KA, MH)
    india_holidays = holidays.India(years=year)
    
    # Placeholders for our key dynamic festivals
    festivals = {
        "diwali": "October/November (Dynamic calculation pending)",
        "durga_puja": "October",
        "eid_fitr": "March/April",
        "akshaya_tritiya": "April/May",
        "ugadi": "March/April"
    }
    
    # Read through calculated dates to locate exact months
    for date_obj, name in sorted(india_holidays.items()):
        name_lower = name.lower()
        month_name = date_obj.strftime("%B")
        day_num = date_obj.strftime("%d")
        
        if "diwali" in name_lower or "deepavali" in name_lower:
            festivals["diwali"] = f"{month_name} (Diwali falls on {month_name} {day_num} in {year})"
        elif "durga" in name_lower or "dussehra" in name_lower or "vijayadashami" in name_lower:
            festivals["durga_puja"] = f"{month_name} (Durga Puja / Dussehra falls in {month_name})"
        elif "ramzan" in name_lower or "id-ul-fitr" in name_lower or "eid" in name_lower:
            # Capturing the main Eid-ul-Fitr
            if "adha" not in name_lower:
                festivals["eid_fitr"] = f"{month_name} (Eid al-Fitr is celebrated in {month_name})"
        elif "ugadi" in name_lower or "gudi padwa" in name_lower:
            festivals["ugadi"] = f"{month_name} (Ugadi/Gudi Padwa falls in {month_name})"
        elif "akshaya" in name_lower:
            festivals["akshaya_tritiya"] = f"{month_name} (Akshaya Tritiya falls in {month_name})"
            
    return festivals
def generate_data_explanation(user_query, df_summary, client_instance):
    """
    Feeds the final processed data back to the LLM to generate 
    a clean, executive insight summary mapping month-wise spikes
    to major Indian religious and cultural festivals dynamically computed via the holidays library.
    """
    try:
        cleaned_df = df_summary.copy()
        cols_to_drop = [col for col in cleaned_df.columns if cleaned_df[col].dtype == 'bool']
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        data_string = cleaned_df.to_string()
    except:
        data_string = str(df_summary)
        
    # 1. Resolve active year dynamically to fetch the correct shifting festivals    
    query_year = 2026  # Default fallback
    if max_database_date:
        try:
            query_year = pd.to_datetime(max_database_date).year
        except:
            pass     
    for year_candidate in range(2024, 2030):
        if str(year_candidate) in str(user_query):
            query_year = year_candidate
            break

    # 2. DYNAMICALLY compute shifting festival schedules
    shifts = get_dynamic_indian_festivals(query_year)

    # 3. Build the Context-Aware Festival Guide
    festival_context_guide = f"""
    ### MONTH-WISE INDIAN FESTIVAL REFERENCE GUIDE (FOR THE YEAR {query_year}):
    Use this exact chronological calendar to explain revenue spikes or performance anomalies:
    - January: Pongal, Makar Sankranti, Lohri (Significant gifting/retail surge in South & North India)
    - February: Season transitions / Valentine's gifting trends
    - March: Ugadi/Gudi Padwa ({shifts['ugadi'] if 'March' in shifts['ugadi'] else 'Sometimes in March'}), Eid al-Fitr ({shifts['eid_fitr'] if 'March' in shifts['eid_fitr'] else ''})
    - April: Ugadi/Gudi Padwa ({shifts['ugadi'] if 'April' in shifts['ugadi'] else ''}), Eid al-Fitr ({shifts['eid_fitr'] if 'April' in shifts['eid_fitr'] else ''}), Akshaya Tritiya ({shifts['akshaya_tritiya'] if 'April' in shifts['akshaya_tritiya'] else ''}) (Highly auspicious gold/watch purchasing)
    - May: Akshaya Tritiya ({shifts['akshaya_tritiya'] if 'May' in shifts['akshaya_tritiya'] else ''}) (Major watch/jewelry buying period), Wedding season demand
    - June/July: Normal trading months (No major national festival peaks)
    - August/September: Raksha Bandhan, Onam, Ganesh Chaturthi (Festive build-up and gifting)
    - September/October: Durga Puja / Dussehra ({shifts['durga_puja']})
    - October/November: Dhanteras & Diwali ({shifts['diwali']}) (The absolute peak national purchasing and watch buying window of the year)
    - December: Christmas, Year-end wedding peak purchases
    """
    
    # Resolve active month and year to feed context to the AI
    active_month_context = ""
    if max_database_date:
        try:
            dt = pd.to_datetime(max_database_date)
            active_month_context = f"Note: The current active month in progress is {dt.strftime('%B %Y')} (up to {dt.strftime('%Y-%m-%d')})."
        except:
            pass
    
    explanation_prompt = f"""
    You are a principal business intelligence analyst at Titan. 
    Analyze this data matrix compiled directly from Redshift to address the user request: "{user_query}"
    
    ### DATA RECOVERY MATRIX:
    {data_string}
    
    ### ANALYTICAL CONTEXT:
    {active_month_context}
    {festival_context_guide}
    
    ### INSTRUCTIONS:
    Provide a crisp, clear 3-bullet point executive explanation of this data for business leaders.
    - Point 1: Highlight the clear leader, highest peak performance, or maximum metrics item.
    - Point 2: Note any clear downward performance trends, minimal contributions, or gaps.
    - Point 3: State a constructive takeaway observation using scale indicators (e.g. Lakhs, Crores, or Percentages) based on the figures provided.
    
    STRICT BUSINESS INTELLIGENCE 
    - SPECIFIC FESTIVAL ATTRIBUTION: If you notice a high peak, revenue surge, or anomaly in a specific month, correlate it with the 'MONTH-WISE INDIAN FESTIVAL REFERENCE GUIDE (FOR THE YEAR {query_year})'. You MUST explicitly name the correct shifting festival (e.g., "driven by Diwali / Dhanteras purchases in {shifts['diwali'].split(' ')[0]}" or "due to Akshaya Tritiya in {shifts['akshaya_tritiya']}") instead of calling it "seasonal activity" or "holiday trends".
    - NO INTRODUCTIONS OR PREAMBLES: Start your response directly with the bullet points. Do NOT include any introductory greetings, meta-commentary, or setup sentences like "As the Principal Business Intelligence Analyst...", "Here is the executive summary...", or "Based on the data...".
    - NO INCOMPLETE-MONTH BIAS: The current active month in progress is an incomplete period. Do NOT treat lower sales in this month as an "anomaly", "downward trend", "gap", or "drop". When identifying historical peaks, anomalies, or performance drops, ONLY consider fully completed historical months.
    - Keep it professional and accurate to the numbers.
    - TYPO CORRECTION: If the user query contains spelling mistakes, correct them silently in your mind. Use the correct, standardized brand names (e.g., Tanishq, Sonata, Mia) and metric names (e.g., Sales, Contribution) in your bullet points and chart titles.
    STRICT COMPLIANCE RULES:
    - Do NOT include structural programming vocabulary like 'dataframe', 'column names', 'NaN', 'rows', or 'SQL query'.
    - Do NOT include raw Boolean terms like 'True', 'False', or comparison flag results anywhere in the final narrative. Keep it strictly focused on business values and labels.
    """
    try:
        response = client_instance.chat.completions.create(
            model="gemini-3.1-flash-lite",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"💡 Contextual Analysis: Summary generated based on real-time parameters. Error fetching text explanation: {exc}"

# On App Startup: Just pull a lightweight preview schema safely to get available_columns
try:
    if not redshift_api_key:
        st.error("❌ 'REDSHIFT_API_KEY' is missing from Secrets!")
        st.stop()
    df_raw = run_redshift_query("SELECT * FROM sem_pu_wtch.mv_tgt_wtch_etp_mendix_cs_metrics_bi_storewise LIMIT 5", redshift_api_key)  
    available_columns = list(df_raw.columns)  
         
    with st.sidebar:  
        st.success("📊 Redshift Connected Successfully")
except Exception as e:
    st.error(f"Could not connect to Redshift. Error: {e}")
    st.stop()

# --- DISPLAY RUNNING CHAT TIMELINE (Oldest at top, Newest at bottom) ---
for interaction in st.session_state['interaction_history']:
    st.write("---")
    # User Query  
    with st.chat_message("user"):  
        st.write(interaction['query'])  
                 
    # Assistant Response  
    with st.chat_message("assistant"):  
        if interaction.get('generated_sql'):  
            with st.expander("📄 View Generated SQL Query", expanded=False):  
                st.code(interaction['generated_sql'], language="sql")  
                     
        if interaction.get('is_error'):
            st.error(f"⚠️ Error details: {interaction.get('error_msg')}")
            with st.expander("🔍 Click to view Traceback Context"):
                st.code(interaction.get('traceback'), language="python")
        else:
            if interaction['output_type'] == "TABLE":  
                if interaction['table_df'] is not None:  
                    st.dataframe(interaction['table_df'], use_container_width=True)  
            else:  
                img_path = interaction['img_path']  
                if img_path and os.path.exists(img_path):  
                    st.image(img_path, width=400)
           
            # Display Modern Highlight Box for generated Business Insights
            if interaction.get('explanation'):
                st.markdown("### 📊 Explanation")
                st.info(interaction['explanation'])

# --- CHATBOT CHAT INPUT INTERFACE AT BOTTOM ---
user_query = st.chat_input("Ask anything about your data... (e.g., Show 2025 MTD sales)")

if user_query:
    # ==========================================
    # 🌟 NEW TEXT ROUTER FOR GREETINGS & GENERAL QUESTIONS 🌟
    # ==========================================
    classification_prompt = f"""
    You are an AI assistant for an enterprise data warehouse chatbot.
    Analyze the user's query: "{user_query}"
    
    If the query is a greeting (like 'hi', 'hello', 'hey') or a casual/general question (like 'how are you', 'what can you do?'), 
    respond with a friendly, direct conversational text reply guiding them to ask about data, and start your response with the prefix 'CHAT_REPLY: '.
    
    Example response for a greeting: "CHAT_REPLY: Hello! I am your Enterprise Intelligence Engine. How can I help you with your data analytics today?"
    
    Otherwise, if it is a proper question asking for sales, metrics, charts, tables, or trends, 
    respond with exactly: "PROCEED_TO_DATA".
    """
    
    try:
        class_response = client.chat.completions.create(
            model="gemini-3.1-flash-lite",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()
    except Exception as e:
        class_response = "PROCEED_TO_DATA" # Fallback to prevent breaking on API errors
        
    if class_response.startswith("CHAT_REPLY:"):
        # Strip out the prefix to get just the clean text answer
        clean_reply = class_response.replace("CHAT_REPLY:", "").strip()
        
        # Save it as a regular text breakdown using the explanation modern highlight box style
        st.session_state['interaction_history'].append({  
            'query': user_query,  
            'output_type': 'TABLE', # Triggers the layout structure without throwing chart logic
            'table_df': None,       # Leaves data visualization completely blank
            'img_path': None,  
            'generated_sql': None,
            'explanation': clean_reply # Puts your friendly chat reply in the modern highlight box
        })
        st.rerun()

    active_sql = None
    try:
        max_date_df = run_redshift_query("SELECT MAX(date) as max_date FROM sem_pu_wtch.mv_tgt_wtch_etp_mendix_cs_metrics_bi_storewise", redshift_api_key)
        if not max_date_df.empty and max_date_df['max_date'].iloc[0] is not None:
            max_dt = pd.to_datetime(max_date_df['max_date'].iloc[0])
            max_database_date = max_dt.strftime('%Y-%m-%d')
        else:
            max_dt = pd.to_datetime(datetime.date.today())
            max_database_date = max_dt.strftime('%Y-%m-%d')
    except:
        max_dt = pd.to_datetime(datetime.date.today())
        max_database_date = max_dt.strftime('%Y-%m-%d')

    # Extract dynamic properties of the max database checkpoint date
    current_month_str = f"{max_dt.month:02d}"
    current_day_str = f"{max_dt.day:02d}"

    # Extract specific year text explicitly if available in user query
    asked_years = [int(word) for word in user_query.split() if word.isdigit() and len(word) == 4]
    target_year_context = asked_years[0] if asked_years else int(max_database_date.split('-')[0])
 
    timestamp = int(time.time())  
    current_chart_filename = os.path.join(HISTORY_FOLDER, f"chart_{timestamp}.png").replace("\\", "/")  

    normalized_query = user_query.lower()
    normalized_query = normalized_query.replace("customer count", "cust")
    normalized_query = normalized_query.replace("customer base", "cust")
    normalized_query = normalized_query.replace("customers", "cust")
    normalized_query = normalized_query.replace("customer", "cust")

    prompt = f"""  
    You are an expert data analyst. Your job is to output a Python Pandas aggregation snippet AND the plotting code to address the user's request.  
         
    CRITICAL DATABASE METADATA:  
    - Table Name: `sem_pu_wtch.mv_tgt_wtch_etp_mendix_cs_metrics_bi_storewise`  
    - Available Columns: {available_columns}  
         
    User Query: "{user_query}"  
   
    ANCHOR REFERENCE PARAMETERS:
    - Target Year Context: {target_year_context}
    - Current Active Month Index: {current_month_str}
    - Current Active Day Index: {current_day_str}
    - Global Database Max Date Line: '{max_database_date}'
    STRICT TIME-SERIES
    1. MONTHLY TRENDS: When the user asks for month-on-month trends or monthly details, always use `DATE_TRUNC('month', date) AS month` or `DATE_TRUNC('month', date)::date AS month` in the SQL query.
    2. PANDAS MONTH CONVERSION: In the generated Python snippet, explicitly cast the `'month'` column using:
       `df['month'] = pd.to_datetime(df['month']).dt.strftime('%b %Y')` to prevent metric identification failures during plotting.

    1. STRICT BUSINESS LOGIC, TIME LABELS, & FISCAL PERIOD CALCULATIONS:
       1. STRICT TIME BOUNDARY HIERARCHY (APPLIES TO REGION-WISE, CHANNEL-WISE, STORE-WISE, OR BRAND-WISE GROUPS):
       - RULE A (MTD MANDATORY FOR ALL SALES/REVENUE): If the query is about "sales" or "revenue" (including regional views, e.g., "region wise sales" or "revenue by region"), you MUST default strictly to the Month-to-Date (MTD) window. Do NOT use Fiscal Year-to-Date (FYTD) for these requests.
         * MTD SQL filter limit: `date >= '{target_year_context}-{current_month_str}-01' AND date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
       - RULE B (FYTD FOR CONTRIBUTION AND GROWTH ONLY): You MUST ONLY use the Fiscal Year-to-Date (FYTD) time frame if the user explicitly asks for "contribution", "share", "growth", or "YoY".
         * FYTD SQL filter limit: `date >= '{target_year_context}-04-01' AND date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
       - RULE C: The custom mathematical logic for "Service Sale" and "Building Sale" bypasses both Rules A and B and must strictly use its custom column calculation formula.
       2.  FULL YEAR OR HISTORICAL YEAR DEFINITION: 
       - If the query mentions 'full year', 'fiscal year', or specifies a historical year apart from the current active year (e.g., 'sale 2025' or '2024 performance') without explicit MTD boundaries, you MUST assume the custom Indian Fiscal Year layout starting from April of that target context year through March of the consecutive year.
       - Example: For year context 2025, use exact date rules: `date >= '2025-04-01' AND date <= '2026-03-31'`.
    2. STRICT MONTH-TO-DATE (MTD) DEFAULT TRIGGER, SALE AND REVENUE TIME BOUNDARY:
       - If the user query contains general terms like 'sales', 'revenue', 'sales revenue', or asks about performance without specifying a strict month, you MUST default your filter rules to the CURRENT Month-to-date (MTD context)
       - You MUST only default to a Month-to-Date (MTD) window context if the user query explicitly contains the term 'mtd', 'this month', or refers to a specific current active month name (e.g., 'July sales').
       - MTD WINDOW LOGIC: Must always start from day one of the active month line to the exact matching checkpoint day criteria.
       - Example: For year context {target_year_context}, use exact parameters: `date >= '{target_year_context}-{current_month_str}-01' AND date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
    3. YTD WINDOW LOGIC: Starts strictly from April 1st of the fiscal target year context (`'{target_year_context}-04-01'`) and matches up perfectly until the identical maximum relative target day threshold is hit: `date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
    4. FISCAL QUARTER DEFINITIONS WITH IDENTICAL RELATIVE MAX DAY CLIPPING RULES:
       - 'Q1' / 'Quarter 1': Range boundary strictly between `'{target_year_context}-04-01'` and `'{target_year_context}-06-30'`. If the active maximum timeline window lands inside this range framework, clip it cleanly using: `date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
       - 'Q2' / 'Quarter 2': Range boundary strictly between `'{target_year_context}-07-01'` and `'{target_year_context}-09-30'`. If the active maximum timeline window lands inside this range framework, clip it cleanly using: `date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
       - 'Q3' / 'Quarter 3': Range boundary strictly between `'{target_year_context}-10-01'` and `'{target_year_context}-12-31'`. If the active maximum timeline window lands inside this range framework, clip it cleanly using: `date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
       - 'Q4' / 'Quarter 4': Range boundary strictly between `'{target_year_context}-01-01'` and `'{target_year_context}-03-31'`. If the active maximum timeline window lands inside this range framework, clip it cleanly using: `date <= '{target_year_context}-{current_month_str}-{current_day_str}'`.
    5. TIMELINE REVENUE RESTRICTIONS: For any baseline relative summaries ('MTD', 'QTD', 'YTD', 'Q1', 'Q2', etc.), strictly aggregate using `retail_revenue` unless custom sale formulas are triggered below.
    6. DETAILED ROW OVERRIDES (TIMEOUT PREVENTION): If the user query contains 'detail', 'details', 'list', or asks for individual records, do NOT run GROUP BY operations or aggregate SUM functions. Generate a clean SELECT statement choosing key column categories filtering data down using a strict 'LIMIT 100' constraint.
    7. USER TYPO RESILIENCE: You must expect and automatically correct spelling mistakes or typos in the user query (e.g., treat "tanshiq" as "Tanishq", "fastrak" as "Fastrack", "sles" as "sales"). Always map misspelled inputs to the exact, correct database table columns, values, and chart labels.
    
    STRICT CONTRIBUTION AND GROWTH FORMULA RULES:
    1. CONTRIBUTION FORMULA: If the user asks for "contribution" or "share", calculate it as:
      `(Segment Sales / Total Sales) * 100` to represent the percentage contribution.
    - You must align the "Total Sales" denominator to the exact same fiscal time window (FYTD, MTD, or Historical Year) applied to the "Segment Sales" numerator.
    2. GROWTH FORMULA: If the user asks for "growth", calculate the Year-over-Year (YoY) percentage change using the formula:
      `((This_Year_Sales - Last_Year_Sales) / Last_Year_Sales) * 100`
    - "This Year" refers strictly to the current fiscal year performance up until the maximum available database date (`'{max_database_date}'`).
    - "Last Year" refers to the exact matching corresponding period of the previous fiscal year.
    - Always evaluate both periods on the same fiscal timeline baseline to maintain analytical consistency.
    3.CONTRIBUTION & GROWTH FORMATTING REQUIREMENT: Any contribution, market share, or growth metrics MUST be calculated as percentages. When plotting, the text labels on top of bars must append a '%' sign suffix.
    STRICT FORMATTING RULES:
    1. HISTORICAL FORMATTING RULE: If user query specifies a past/future year(e.g.,'2025','2024') or past/future month (e.g., 'may2025','jan 2025'), you MUST NOT INCLUDE 'MTD', 'QTD', 'YTD', 'Month-to-date' or ' Year-to-date' in the chart tittle ubder any circumstances.
    - for example,if query is sales "sales 2025" or "mtd 2025", the tittle MUST ve formatted simple as " Revenue 2025" or "Sales 2025".
    2. CURRENT ACTIVE PERUOD ONLY: You may only include terms like 'MTD', 'QTD', 'YTD' in the chart title if the requestis strictly for current active month and year context(i.e., '{max_database_date}').
    CUSTOMER COLUMN MAPPING RULE:
    - The dataset does not contain a column named customer.
    - Any customer-related query must map to the correct `*_cust` field.
    - Use:
      service_cust, retail_cust, wdc_cust, others_cust, no_discount_cust, str_cust
    STRICT BUSINESS FORMULA DEFINITIONS:
    - If user asks for 'Service Sale' or 'Service Sales': You MUST calculate it exactly as: `sum(service_revenue)+ sum(str) + sum(wdc) + sum(others) + sum(no_discount)`.
    - If user asks for 'Building Sale' or 'Building Sales': You MUST calculate it exactly as: `sum(service_revenue) + sum(retail_revenue)`.
    STRICT BRAND & ABBREVIATION ISOLATION RULES:
    - HELIOS / Helios -> Match using strict exact uppercase values: `(UPPER(channel) = 'HLS')`.
    - FASTRACK / Fastrack -> Match using strict exact uppercase values: `(UPPER(channel) = 'FTS')`.
    - TITAN WORLD / Titan World / TW -> Match using strict exact uppercase values: `(UPPER(channel) = 'TW')`.
    STRICT VALUE & COLUMN MAPPING RULES:
    - If the user asks for 'Mita', search checking case-insensitive wildcard configurations: `WHERE (UPPER(rbm_name) LIKE '%MITA%' OR UPPER(abm_name) LIKE '%MITA%' OR UPPER(abm_name_csm) LIKE '%MITA%')`
    - Apply identical broad matching checking rules for 'Upal': `WHERE (UPPER(rbm_name) LIKE '%UPAL%' OR UPPER(abm_name) LIKE '%UPAL%')`
    STRICT RUNTIME & PLOTTING ERROR MITIGATION RULES:
    - CRITICAL: Always explicitly convert your numeric/metric plotting columns to float using `pd.to_numeric(df[col], errors='coerce')` before passing data to matplotlib to avoid string/categorical axis scaling bugs.
    - MULTI-CHART COMPATIBILITY: Do not limit yourself strictly to vertical bars. You can use line plots (`plt.plot`), pie charts (`plt.pie`), or bar charts (`plt.bar`/`plt.barh`) depending on the context of the user request.
    - TYPE COMPATIBILITY FOR PLOTTING: Always cast textual or categorical columns explicitly to a string matrix using `df[col] = df[col].astype(str)` before sending data directly to axes.
         
    STRICT OUTPUT STRUCTURING RULE:  
    You must output your entire response in exactly two parts separated by a unique delimiter line: '---PLOT_CODE_START---'.  
   
    Do NOT omit this token block or change its text formatting.
    On the very first line of your response, print your custom SQL statement inside a comment block formatted exactly like this:  
    # NEW_SQL: <YOUR_GENERATED_SQL_QUERY_HERE>  
         
    CRITICAL TABLE DETECTION: If the user wants a table, grid, or text list, or if the query contains the word 'detail'/'details', write the comment '# OUTPUT_TYPE: TABLE' on the line directly under the SQL comment block. Otherwise, write '# OUTPUT_TYPE: CHART'.  
   
    Everything after '---PLOT_CODE_START---' MUST BE PURE, RUNNABLE PYTHON MATPLOTLIB CODE.  
    Assume whatever data your custom SQL query pulls will be provided downstream as a DataFrame named `df_raw`. Build your summary dataframe named `df` directly from `df_raw`.  
   
    STRICT VISUALIZATION LABELLING & MULTI-COLUMN CODE RULES:
    - Write robust text label logic inline that formats values into Lakh/Cr or percentages. It must check if container elements are bar objects before calculating coordinates:
    ```python
    for ax in plt.gcf().axes:
        if hasattr(ax, 'containers') and ax.containers:
            for container in ax.containers:
                for bar in container:
                    try:
                        yval = bar.get_height() if bar.get_height() != 1.0 else bar.get_width()
                        if hasattr(bar, 'get_width') and bar.get_width() == 1.0:
                            yval = bar.get_height()
                        if pd.isna(yval) or yval == 0: continue
                        if "growth" in "{user_query.lower()}" or "contribution" in "{user_query.lower()}" or "share" in "{user_query.lower()}":
                            lbl = f"{{yval:.2f}}%"
                        elif yval >= 10000000:
                            lbl = f"{{yval / 10000000:.2f}} Cr"
                        elif yval >= 100000:
                            lbl = f"{{yval / 100000:.2f}} Lakh"
                        else:
                            lbl = f"{{yval:.2f}}"
                        if bar.get_height() != 1.0 and bar.get_width() != 1.0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), lbl, ha='center', va='bottom', fontsize=8, fontweight='bold')
                    except: pass
    ```
    - Save graph to: `plt.savefig("{current_chart_filename}", bbox_inches="tight")`.  
    """  
           
    with st.spinner("Processing dashboard data filters..."):  
        try:  
            response = client.chat.completions.create(  
                model="gemini-3.1-flash-lite",  
                messages=[{"role": "user", "content": prompt}],  
                temperature=0.1  
            )  
                     
            raw_text = response.choices[0].message.content  
                     
            if "---PLOT_CODE_START---" in raw_text:  
                pandas_logic, execution_content = raw_text.split("---PLOT_CODE_START---")  
                pandas_logic = pandas_logic.strip()  
                execution_content = execution_content.strip()  
                                 
                # Intercept fallback SQL value  
                active_sql = f"SELECT date, region, retail_revenue FROM sem_pu_wtch.mv_tgt_wtch_etp_mendix_cs_metrics_bi_storewise WHERE date <= '{max_database_date}' LIMIT 10"  
                is_table = False  
                                 
                for line in pandas_logic.split("\n"):  
                    if line.startswith("# NEW_SQL:"):  
                        active_sql = line.replace("# NEW_SQL:", "").strip()  
                    if line.startswith("# OUTPUT_TYPE: TABLE"):  
                        is_table = True  
                                 
                # Check for table keyword overrides  
                if any(w in user_query.lower() for w in ["table", "grid", "detail", "details", "list"]):  
                    is_table = True  
                                 
                df_raw = run_redshift_query(active_sql, redshift_api_key)  
                                             
                # Set up isolated workspace dictionary environment context
                shared_context = {'df_raw': df_raw, 'pd': pd, 'plt': plt, 'df': None}  
                               
                if is_table:  
                    df = df_raw.copy()  
                    shared_context['df'] = df
                else:  
                    # Run transformations logic
                    exec(pandas_logic, globals(), shared_context)  
                   
                    # Run graphing logic immediately to let chart definitions alter metrics layout
                    plt.close('all')  
                    exec(execution_content, globals(), shared_context)
                   
                    # FIX: Extract mutating 'df' from the context context window AFTER code components evaluate
                    df = shared_context.get('df')  
               
                # --- STAGE 2: GENERATE EXECUTIVE SUMMARY ---
                if df is not None and not df.empty:
                    with st.spinner("Analyzing executive business insights..."):
                        insights_narrative = generate_data_explanation(user_query, df.head(25), client)
                else:
                    insights_narrative = "💡 Analysis Note: The requested data scope returned no matching transaction records."
                                 
                # Handle Table outputs  
                if is_table:  
                    df_display = df.reset_index() if (df is not None and (isinstance(df.index, pd.MultiIndex) or df.index.name is not None)) else (df.copy() if df is not None else pd.DataFrame())  
                                         
                    st.session_state['interaction_history'].append({  
                        'query': user_query,  
                        'output_type': 'TABLE',  
                        'table_df': df_display,  
                        'img_path': None,  
                        'generated_sql': active_sql,
                        'explanation': insights_narrative
                    })  
                                 
                # Handle Chart outputs  
                else:                                              
                    if os.path.exists(current_chart_filename):  
                        st.session_state['interaction_history'].append({  
                            'query': user_query,  
                            'output_type': 'CHART',  
                            'table_df': None,  
                            'img_path': current_chart_filename,  
                            'generated_sql': active_sql,
                            'explanation': insights_narrative
                        })  
            else:  
                st.error("Formatting structure parsing mismatch. Please re-phrase your request.")  
                                         
            st.rerun()  
                                 
        except Exception as e:  
            error_str = str(e).lower()  
            trace_str = traceback.format_exc().lower()  
            friendly_message = "An unexpected error occurred while processing your data request."  
                     
            # 1. Identify Year-Specific Missing Data Errors  
            if any(str(yr) in user_query for yr in asked_years):  
                missing_year = asked_years[0] if asked_years else "requested"  
                if "keyerror" in trace_str or "empty" in error_str or "none" in error_str:  
                    friendly_message = f"📊 Data for the fiscal year {missing_year} is currently unavailable or missing from the system."  
                     
            # 2. Identify Column/Metric Mismatch Errors  
            elif "keyerror" in trace_str:  
                missing_col = str(e).replace('"', '').replace("'", "")  
                friendly_message = f"🔍 Could not find the metric '{missing_col}' for this selection. Please check if the requested timeframe or filter has valid data."  
                     
            # 3. Identify Database connection timeouts or execution bugs  
            elif "gateway error" in error_str or "timeout" in error_str:  
                friendly_message = "⚡ The database connection timed out. Please try filtering for a smaller date range or specific region."  
                     
            # 4. Fallback default clean message  
            else:  
                friendly_message = f"⚠️ system notice: We couldn't generate this specific view. This usually happens if the filter combination yields no matching records."  
                 
            st.session_state['interaction_history'].append({  
                'query': user_query,  
                'output_type': 'ERROR',  
                'table_df': None,  
                'img_path': None,  
                'generated_sql': active_sql if active_sql else "SQL extraction step failed before execution.",  
                'is_error': True,  
                'error_msg': friendly_message,  
                'traceback': traceback.format_exc(),
                'explanation': None
            })  
            st.rerun()

# --- SIDEBAR ACTION: CLEAR ACTIONS BUTTON ---
with st.sidebar:
    st.write("---")
    if st.button("🗑 Clear History", use_container_width=True):
        for interaction in st.session_state['interaction_history']:
            img_file = interaction.get('img_path')
            if img_file and os.path.exists(img_file):
                try:
                    os.remove(img_file)
                except:
                    pass
        st.session_state['interaction_history'] = []
        plt.close('all')
        st.success("App history cleared!")
        st.rerun()
