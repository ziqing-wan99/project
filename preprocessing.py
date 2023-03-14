import re
import pandas as pd
from datetime import datetime
import nltk
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
from geopy.geocoders import Nominatim

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('ChatGPT.csv', encoding='utf8', low_memory=False)
pd.set_option('display.max_columns', None)


def clean_text(text):
    steps = [convert_to_lowercase,
             remove_urls,
             remove_line_break,
             keep_letter_digit_whitespace,
             remove_stopwords,
             stem]
    for step in steps:
        text = step(text)
    return text


# convert text to lowercase
def convert_to_lowercase(text):
    text = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text)


# remove all urls
def remove_urls(text):
    return re.sub(r'http\S+', '', text)


# replace all line breaks (\r and \n) with spaces
def remove_line_break(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text


# remove any character that is not a letter, digit, or whitespace character
def keep_letter_digit_whitespace(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


# remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in tokens if word not in stop_words])
    return text


# Snowball Stemmer
def stem(text):
    stemmer = SnowballStemmer('english')
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text)]
    return ' '.join(stemmed_text)


# Tweet text filtering
df["Tweet"] = df["Tweet"].astype(str)
df['clean_Tweet'] = pd.Series([clean_text(i) for i in tqdm(df['Tweet'])])
tweets = df["clean_Tweet"].values
tweet_list = []
for i in tweets:
    tweet_list.append(str(i))


# print(tweet_list[:5])
# print(df.head(2))


def categorized_UserFollowers(user_followers_count):
    if float(user_followers_count) < 1000:
        category = "Low"
    elif float(user_followers_count) < 10000:
        category = "Medium"
    elif float(user_followers_count) >= 10000:
        category = "High"
    else:
        category = "Undefined"
    return category


# Categorize UserFollowers
df["UserFollowers"] = df["UserFollowers"].astype(str)
df['categorized_UserFollowers'] = pd.Series([categorized_UserFollowers(i) for i in tqdm(df['UserFollowers'])])
user_followers_levels = df["categorized_UserFollowers"].values
user_followers_level_list = []
for i in user_followers_levels:
    user_followers_level_list.append(str(i))


# print(user_followers_level_list[460:470])
# print(df.head(3))


def categorized_UserCreated(user_created_time):
    format_str = '%Y-%m-%d %H:%M:%S+00:00'
    try:
        datetime.strptime(user_created_time, format_str)
        return datetime.strptime(user_created_time, format_str).year
    except ValueError:
        return "Undefined"


# Convert UserCreated to year
df["UserCreated"] = df["UserCreated"].astype(str)
df['categorized_UserCreated'] = pd.Series([categorized_UserCreated(i) for i in tqdm(df['UserCreated'])])
user_created_year = df["categorized_UserCreated"].values
user_created_year_list = []
for i in user_created_year:
    user_created_year_list.append(str(i))


# print(user_created_year_list[53954])
# print(df.head(1))


def get_user_occupation(user_description):
    # look for occupation-related keywords
    occupation_keywords = ['Accountant', 'Accounting', 'Actor/Actress', 'Administration', 'Advertising', 'Analyst',
                           'Architect', 'Artist', 'Attorney/Lawyer', 'Author', 'Banker/Banking', 'Biologist',
                           'Broker/Brokerage', 'Business/Businessman/Businesswoman', 'CEO/Chief Executive Officer',
                           'CFO/Chief Financial Officer', 'Chemist/Chemical', 'Civil Engineer/Engineering', 'Coach',
                           'Communications', 'Consultant/Consulting', 'Content Creator', 'Copywriter', 'Counselor',
                           'Creative', 'Customer Service', 'Data Analyst', 'Dentist', 'Designer/Design',
                           'Developer/Development', 'Digital Marketer/Marketing', 'Director', 'Doctor/Medical',
                           'E-commerce', 'Economist/Economics', 'Editor', 'Educator/Education', 'Electrician',
                           'Engineer/Engineering', 'Entrepreneur', 'Environmentalist/Environmental',
                           'Event Planner/Planning', 'Executive', 'Fashion/Fashion Designer', 'Film/Filmmaker',
                           'Financial/Fintech', 'Freelance/Freelancer', 'Fundraiser/Fundraising', 'Gamer/Gaming',
                           'Graphic Designer', 'Health/Healthcare', 'HR/Human Resources', 'Illustrator',
                           'Information Technology/IT', 'Interior Designer/Design', 'Investor/Investing',
                           'Journalist/Journalism', 'Lawyer/Legal', 'Lecturer/Lecturing', 'Librarian/Library',
                           'Manager/Management', 'Marketer/Marketing', 'Mechanic/Mechanical', 'Media',
                           'Medical/Medicine', 'Music/Musician', 'Non-profit/Nonprofit', 'Nurse/Nursing',
                           'Nutritionist', 'Operations', 'Optometrist/Optometry', 'Painter/Painting',
                           'Personal Trainer', 'Pharmacist/Pharmacy', 'Photographer/Photography',
                           'Physician/Physicians', 'Pilot/Aviation', 'Planner/Planning', 'Podcaster',
                           'Product Manager/Management', 'Programmer/Programming', 'Project Manager/Management',
                           'Public Relations/PR', 'Publisher/Publishing', 'Real Estate/Real Estate Agent',
                           'Recruiter/Recruiting', 'Researcher/Research', 'Sales/Salesperson', 'Scientist/Science',
                           'Social Media/Social Media Manager', 'Software/Software Engineer',
                           'Speech Therapist/Therapy', 'Sports/Sporting', 'Strategist/Strategy',
                           'Student/Student Affairs', 'Surgeon/Surgery', 'Teacher/Teaching', 'Technician/Technology',
                           'Therapist/Therapy', 'Translator/Translation', 'Travel/Traveling',
                           'Videographer/Videography', 'Virtual Assistant', 'Web Developer/Development',
                           'Writer/Writing']

    new_occupation_keywords = []
    for keyword in occupation_keywords:
        if '/' in keyword:
            new_keywords = keyword.split('/')
            new_occupation_keywords.extend(new_keywords)
        else:
            new_occupation_keywords.append(keyword)
    occupation = ""
    input_words = user_description.split()
    for word in input_words:
        if word in [x.lower() for x in new_occupation_keywords]:
            occupation += word + " "
    return occupation.strip()


def clean_description(text):
    steps = [convert_to_lowercase,
             remove_urls,
             remove_line_break,
             keep_letter_digit_whitespace,
             remove_stopwords]
    for step in steps:
        text = step(text)
    return text


# Get UserOccupation from UserDescription
df["UserDescription"] = df["UserDescription"].astype(str)
df['clean_UserDescription'] = pd.Series([clean_description(i) for i in tqdm(df['UserDescription'])])
df['UserOccupation'] = pd.Series([get_user_occupation(i) for i in tqdm(df['clean_UserDescription'])])

user_occupation = df["clean_UserDescription"].values
user_occupation_list = []
for i in user_occupation:
    user_occupation_list.append(str(i))
# print(user_occupation_list[:30])

user_occupation = df["UserOccupation"].values
user_occupation_list = []
for i in user_occupation:
    user_occupation_list.append(str(i))


# print(user_occupation_list[:30])

def get_country(user_location):
    geolocator = Nominatim(user_agent="cs5344")
    try:
        location = geolocator.geocode(user_location)
        if location is None:
            country = "Undefined"
        else:
            country = location.address.split(", ")[-1]
    except (AttributeError, GeocoderTimedOut):
        country = "Undefined"
    except GeocoderServiceError as e:
        country = "Undefined"
    return country


# lru_cache(maxsize=1000)
#
# df["Location"] = df["Location"].astype(str)
# df['Country'] = pd.Series([get_country(i) for i in tqdm(df['Location'])])
# user_country = df["Country"].values
# user_country_list = []
# for i in user_country:
#     user_country_list.append(str(i))
# print(user_country_list[:50])
# print(df.head(1))

df.to_csv('preprocessed_ChatGPT.csv')
