import logging
import os
import uuid
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine
import Config as cfg

load_dotenv()
LOGGER = logging.getLogger("data_generator")
FILE_NAME = "DataGenerator.py"
DESTINATION_TABLE = "Pratham_tbl_mi_phishing_simulation_data_clean"
server = os.getenv('server')
database = os.getenv('database')

def inject_nulls_strict(df, null_prob=0.0005):
    mask = np.random.rand(*df.shape) < null_prob
    df = df.mask(mask)
    return df

def rand_date(start="2000-01-01", end="2026-01-01"):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return s + pd.to_timedelta(random.randint(0, (e - s).days), unit='d')

def rand_ts():
    d = rand_date()
    return d + pd.to_timedelta(random.randint(0, 86400), unit='s')

def rand_choice(arr):
    return random.choice(list(arr))

def rand_id():
    return str(uuid.uuid4())

def rand_email(fn, ln):
    return f"{fn.lower()}.{ln.lower()}@barclays.com"

# Load configurations from Config module
cities = list(cfg.city_to_zone.keys())
grades = list(cfg.grade_mapping.keys())
coo_areas = list(cfg.coo_area_mapping.keys())
event_types = list(cfg.target_function.keys())
email_keywords = [kw for v in cfg.email_config.values() for kw in v]
business_keywords = [kw for v in cfg.business_area_config.values() for kw in v]

first_names = ["Aaron", "George", "Neha", "Priyanka", "Shivani"]
last_names = ["Patil", "Singh", "Khanna", "Johnson", "Smith"]
campaign = ["PhishingSimulation_February2026", "PhishingSimulation_January2026", "PhishingSimulation_March2026"]

# Senior people generally have longer tenure
def generate_hire_date(grade):
    if grade in ['md', 'd']:
        years = random.randint(10, 25)
    elif grade == 'vp':
        years = random.randint(6, 15)
    elif grade == 'avp':
        years = random.randint(3, 10)
    else:
        years = random.randint(0, 8)
    return pd.Timestamp.now() - pd.DateOffset(years=years)

def generate_subject(category):
    category = random.choice(['credential', 'financial', 'urgency', 'generic', 'social'])
    if category == 'credential':
        return random.choice(['password reset required', 'verify your account', 'secure login required'])
    elif category == 'financial':
        return random.choice(['invoice payment pending', 'wire transfer approval', 'expense reimbursement'])
    elif category == 'urgency':
        return random.choice(['urgent action required', 'account suspended', 'critical system update'])
    elif category == 'social':
        return random.choice(['shared document', 'please review attachment', 'confidential file'])
    else:
        return random.choice(['service notification', 'system maintenance', 'policy update'])

def generate_event(subject, grade, is_hugs):
    ### Base probabilities
    p_click = 0.05
    p_report = 0.10
    
    ### Credential emails
    if any(x in subject for x in ['password', 'account', 'login', 'verify']):
        p_click += 0.20
        
    ### Financial emails
    if any(x in subject for x in ['invoice', 'payment', 'transfer']):
        p_click += 0.15
        
    ### Urgent emails
    if any(x in subject for x in ['urgent', 'critical', 'suspended']):
        p_click += 0.10
        
    ### Senior users report more
    if grade in ['vp', 'd', 'md']:
        p_report += 0.10
        
    ### HUG users report more
    if is_hugs == 'Yes':
        p_report += 0.20
        
    ### Normalize
    p_no_action = max(1 - p_click - p_report, 0.05)
    result = np.random.choice(['no action', 'clicked link', 'reported'], p=[p_no_action, p_click, p_report])
    return result

def generate_dataset(n=1000, null_prob=0.005, times=3):
    LOGGER.info("generate_dataset start | n=%s null_prob=%s times=%s", n, null_prob, times)
    rows = []
    ### Generate unique users
    users = []
    for _ in range(int(n / times)):
        fn = rand_choice(first_names)
        ln = rand_choice(last_names)
        grade = np.random.choice(['ba1', 'ba2', 'ba3', 'ba4', 'avp', 'vp', 'd', 'md'], p=[0.15, 0.20, 0.20, 0.15, 0.12, 0.10, 0.05, 0.03])
        hugs = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])
        brid = rand_id()[:10]
        users.append({
            'firstname': fn, 
            'lastname': ln, 
            'grade': grade, 
            'is_hugs': hugs, 
            'hire_date': generate_hire_date(grade),
            'brid': brid
        })
        
    ### Generate multiple phishing events per user
    for user in users:
        for _ in range(times):
            sent_ts = rand_ts()
            city = rand_choice(cities)
            coo_area = rand_choice(coo_areas)
            subject = generate_subject(rand_choice(event_types))
            event_type = generate_event(subject, user['grade'], user['is_hugs'])
            ba = random.sample(business_keywords, 5)
            cpng = rand_choice(campaign)
            
            row = {
                'reportingdate': str(sent_ts.date()),
                'userfirstname': user['firstname'],
                'userlastname': user['lastname'],
                'useremailaddress': rand_email(user['firstname'], user['lastname']),
                'useractiveflag': 'True',
                'userdeleteddate': 'None',
                'senttimestamp': sent_ts,
                'senttimestamp_minutes': random.randint(1, 1500),
                'eventtype': event_type,
                'campaignname': cpng,
                'autoenrollment': 'False',
                'campaignstartdate': sent_ts,
                'campaigndenddate': sent_ts + timedelta(days=10),
                'campaigntype': 'Drive By',
                'campaignstatus': 'Completed',
                'templatename': 'Generated Template',
                'templatesubject': subject,
                'isattachmentsarchived': 'False',
                'sso_id': rand_id(),
                ### Same user every time
                'usertags-BRID': user['brid'],
                'usertags-Location': city,
                'usertags-Azure UPN': rand_email(user['firstname'], user['lastname']),
                'usertags-Date Added': rand_date().isoformat(),
                'usertags-Department': rand_choice(business_keywords),
                'usertags-Business Unit': 'Barclays Services LLC',
                'usertags-On-Premises Domain Name': 'INTRANET.BARCAPINT.COM',
                'usertags-On-Premises Extension Attribute 5': rand_id(),
                'usertags-On-Premises Extension Attribute 6': rand_email(user['firstname'], user['lastname']),
                'businessarea1': ba[0],
                'businessarea2': ba[1],
                'businessarea3': ba[2],
                'businessarea4': ba[3],
                'businessarea5': ba[4],
                'coo': rand_choice(coo_areas),
                'region': rand_choice(['APAC', 'UK', 'Americas']),
                'country': rand_choice(['India', 'UK', 'USA', 'Philippines']),
                'CISO': rand_choice(['Colleen Rose', 'Andy Piper']),
                'legal_entity': 'UNKNOWN',
                'bu1': None,
                ### Same HUG status every time
                'is_hugs': user['is_hugs'],
                'loaddatetime': datetime.now(),
                ### Same grade every time
                'corporate_grade': user['grade'],
                ### Same hire date every time
                'LocalHireRehireDate': user['hire_date'],
                'city': city,
                'employee_type': rand_choice(['Permanent/Regular', 'Contract']),
                'COO_Area': coo_area,
                'COO': rand_choice(['Manager A', 'Manager B']),
                'proofpoint_brid': user['brid']
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df = inject_nulls_strict(df, null_prob)
    return df

def save_to_mssql(df):
    try:
        engine = create_engine(f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")
        # Replace NaN with NULL
        df = df.where(pd.notnull(df), None)
        df.to_sql(name=DESTINATION_TABLE, con=engine, if_exists='replace', index=False)
        print(f"\n[{FILE_NAME}]: Data saved successfully!")
    except Exception as e:
        print(f"\n[{FILE_NAME}]: Error -> {e}")

if __name__ == '__main__':
    df = generate_dataset(n=100000, null_prob=0.02)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nNULL %:\n", (df.isnull().mean() * 100).round(2))
    save_to_mssql(df)