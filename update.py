import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gradio as gr
import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import traceback
from selenium.common.exceptions import TimeoutException, WebDriverException
import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('course_finder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for caching
cached_courses = None
last_cache_time = 0
CACHE_DURATION = 3600  # Cache for 1 hour
MAX_RETRIES = 3
RETRY_DELAY = 2

# Initialize the model globally
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully initialized the sentence transformer model")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    model = None

def setup_driver():
    """Set up and return a configured Chrome WebDriver"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        
        # Use webdriver_manager to handle driver installation
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        return None

def scroll_to_bottom(driver):
    """Scroll to bottom of page to load more content"""
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        retry_count = 0
        max_scrolls = 10  # Increased from 3 to 10
        while retry_count < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(RETRY_DELAY)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                retry_count += 1
            else:
                retry_count = 0
            last_height = new_height
            if retry_count >= 3:  # If no new content after 3 attempts, break
                break
    except Exception as e:
        logger.error(f"Error while scrolling: {e}")

def safe_get_url(driver, url, max_retries=3):
    """Safely get URL with retries"""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            return True
        except Exception as e:
            logger.error(f"Error loading URL {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(RETRY_DELAY)
    return False

def extract_course_info(card, platform):
    """Extract course information from a card element"""
    try:
        # Find title
        title_tag = card.find(['h2', 'h3', 'h4'], class_=lambda x: x and ('title' in x.lower() or 'heading' in x.lower() or 'name' in x.lower()))
        if not title_tag:
            return None
        title = title_tag.text.strip()
        
        # Find link
        link = card.find('a', href=True)
        if not link:
            return None
        
        course_link = link['href']
        if not course_link.startswith('http'):
            if platform == 'Coursera':
                course_link = "https://www.coursera.org" + course_link
            elif platform == 'Udemy':
                course_link = "https://www.udemy.com" + course_link
            elif platform == 'Analytics Vidhya':
                course_link = "https://courses.analyticsvidhya.com" + course_link
        
        # Find image
        img = card.find('img')
        image_url = img['src'] if img and img.get('src') else ""
        
        return {
            'title': title,
            'image_url': image_url,
            'course_link': course_link,
            'platform': platform
        }
    except Exception as e:
        logger.error(f"Error extracting course info: {e}")
        return None

def scrape_platform(url, platform, card_selectors):
    """Generic function to scrape courses from a platform"""
    courses = []
    driver = None
    
    try:
        driver = setup_driver()
        if not driver:
            return courses
        
        if not safe_get_url(driver, url):
            return courses
        
        # Scroll multiple times to load more content
        for _ in range(5):
            scroll_to_bottom(driver)
            time.sleep(RETRY_DELAY)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Try different selectors
        for selector in card_selectors:
            try:
                # First try with class-based search
                cards = soup.find_all(selector['tag'], class_=lambda x: x and any(c in x.lower() for c in selector['classes']))
                
                # If no results, try with more generic search
                if not cards:
                    cards = soup.find_all(selector['tag'])
                
                for card in cards:
                    # Try to find course information using multiple methods
                    course_info = None
                    
                    # Method 1: Try with title class
                    title_tag = card.find(['h2', 'h3', 'h4'], class_=lambda x: x and ('title' in x.lower() or 'heading' in x.lower() or 'name' in x.lower()))
                    
                    # Method 2: Try with any heading
                    if not title_tag:
                        title_tag = card.find(['h2', 'h3', 'h4'])
                    
                    # Method 3: Try with any text that looks like a title
                    if not title_tag:
                        title_tag = card.find(string=lambda x: x and len(x.strip()) > 10 and len(x.strip()) < 100)
                    
                    if title_tag:
                        title = title_tag.text.strip() if hasattr(title_tag, 'text') else str(title_tag).strip()
                        
                        # Find link
                        link = card.find('a', href=True)
                        if not link:
                            # Try to find link in parent elements
                            parent = card.parent
                            while parent and not link:
                                link = parent.find('a', href=True)
                                parent = parent.parent
                        
                        if link:
                            course_link = link['href']
                            if not course_link.startswith('http'):
                                if platform == 'Coursera':
                                    course_link = "https://www.coursera.org" + course_link
                                elif platform == 'Udemy':
                                    course_link = "https://www.udemy.com" + course_link
                                elif platform == 'Analytics Vidhya':
                                    course_link = "https://courses.analyticsvidhya.com" + course_link
                            
                            # Find image
                            img = card.find('img')
                            if not img:
                                # Try to find image in parent elements
                                parent = card.parent
                                while parent and not img:
                                    img = parent.find('img')
                                    parent = parent.parent
                            
                            image_url = img['src'] if img and img.get('src') else ""
                            
                            course_info = {
                                'title': title,
                                'image_url': image_url,
                                'course_link': course_link,
                                'platform': platform
                            }
                    
                    if course_info and course_info['title'] and course_info['course_link']:
                        courses.append(course_info)
                
                if courses:  # If we found courses with this selector, no need to try others
                    break
                    
            except Exception as e:
                logger.error(f"Error processing selector {selector}: {e}")
                continue
        
        logger.info(f"Successfully scraped {len(courses)} courses from {platform}")
        return courses
        
    except Exception as e:
        logger.error(f"Error scraping {platform}: {e}")
        logger.error(traceback.format_exc())
        return courses
        
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def scrape_coursera():
    """Scrape courses from Coursera"""
    url = "https://www.coursera.org/courses?query=free"
    selectors = [
        {'tag': 'li', 'classes': ['course-card', 'course-item', 'course', 'ais-InfiniteHits-item', 'cds-ProductCard']},
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'cds-ProductCard']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']}
    ]
    return scrape_platform(url, 'Coursera', selectors)

def scrape_udemy():
    """Scrape courses from Udemy"""
    url = "https://www.udemy.com/courses/free/"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'course-card--container']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['course-card--container']}
    ]
    return scrape_platform(url, 'Udemy', selectors)

def scrape_analyticsvidhya():
    """Scrape courses from Analytics Vidhya"""
    url = "https://courses.analyticsvidhya.com/"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'course-card__img-container']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'header', 'classes': ['course-card__img-container']}
    ]
    return scrape_platform(url, 'Analytics Vidhya', selectors)

def scrape_edx():
    """Scrape courses from edX"""
    url = "https://www.edx.org/search?subject=Computer+Science&price=price-free"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'discovery-card']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['discovery-card']}
    ]
    return scrape_platform(url, 'edX', selectors)

def scrape_khanacademy():
    """Scrape courses from Khan Academy"""
    url = "https://www.khanacademy.org/computing"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', '_1q7xw7']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['_1q7xw7']}
    ]
    return scrape_platform(url, 'Khan Academy', selectors)

def scrape_mitocw():
    """Scrape courses from MIT OpenCourseWare"""
    url = "https://ocw.mit.edu/search/?t=Computer%20Science"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'coursePreview']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['coursePreview']}
    ]
    return scrape_platform(url, 'MIT OpenCourseWare', selectors)

def scrape_freecodecamp():
    """Scrape courses from freeCodeCamp"""
    url = "https://www.freecodecamp.org/learn/"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'block']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['block']}
    ]
    return scrape_platform(url, 'freeCodeCamp', selectors)

def scrape_harvard():
    """Scrape courses from Harvard Online Learning"""
    url = "https://online-learning.harvard.edu/catalog?keywords=&subject%5B%5D=2&max_price=&start_date=&availability%5B%5D=1"
    selectors = [
        {'tag': 'div', 'classes': ['course-card', 'course-item', 'course', 'course-card--container']},
        {'tag': 'article', 'classes': ['course-card', 'course-item', 'course']},
        {'tag': 'div', 'classes': ['course-card--container']}
    ]
    return scrape_platform(url, 'Harvard Online Learning', selectors)

def get_all_courses():
    """Get all courses with caching and error handling"""
    global cached_courses, last_cache_time
    current_time = time.time()
    
    # Return cached courses if available and not expired
    if cached_courses is not None and (current_time - last_cache_time) < CACHE_DURATION:
        logger.info("Returning cached courses")
        return cached_courses
    
    logger.info("Starting to scrape courses from all platforms...")
    all_courses = []
    
    try:
        # Scrape from each platform
        platform_scrapers = [
            scrape_coursera,
            scrape_udemy,
            scrape_analyticsvidhya,
            scrape_edx,
            scrape_khanacademy,
            scrape_mitocw,
            scrape_freecodecamp,
            scrape_harvard
        ]
        
        for platform_scraper in platform_scrapers:
            try:
                courses = platform_scraper()
                if courses:
                    all_courses.extend(courses)
                    logger.info(f"Added {len(courses)} courses from {platform_scraper.__name__}")
            except Exception as e:
                logger.error(f"Error in platform scraper {platform_scraper.__name__}: {e}")
                continue

        # Remove duplicates based on title
        unique_courses = []
        seen_titles = set()
        for course in all_courses:
            if course and course['title'] and course['title'] not in seen_titles:
                unique_courses.append(course)
                seen_titles.add(course['title'])
        
        # Update cache
        cached_courses = pd.DataFrame(unique_courses)
        last_cache_time = current_time
        
        logger.info(f"Total unique courses scraped: {len(unique_courses)}")
        return cached_courses
    except Exception as e:
        logger.error(f"Error in get_all_courses: {e}")
        logger.error(traceback.format_exc())
        if cached_courses is not None:
            logger.info("Returning cached courses due to error")
            return cached_courses
        return pd.DataFrame()

def search_courses(query, df):
    """Search courses with error handling"""
    try:
        if df.empty:
            return []
            
        if model is None:
            logger.error("Model not initialized")
            return []
            
        # Encode query and course titles
        query_embedding = model.encode(query, convert_to_tensor=True)
        course_titles = df['title'].tolist()
        course_embeddings = model.encode(course_titles, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]
        top_results = similarities.topk(k=30)  # Keep searching through 30 results to find best matches
        
        seen_courses = set()
        results = []
        
        for idx in top_results.indices:
            try:
                course = df.iloc[idx.item()]
                if course['title'] not in seen_courses:
                    results.append({
                        'title': course['title'],
                        'image_url': course['image_url'],
                        'course_link': course['course_link'],
                        'platform': course['platform'],
                        'score': similarities[idx].item()
                    })
                    seen_courses.add(course['title'])
                    if len(results) >= 8:  # Stop after getting top 8 unique results
                        break
            except Exception as e:
                logger.error(f"Error processing search result: {e}")
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    except Exception as e:
        logger.error(f"Error in search_courses: {e}")
        logger.error(traceback.format_exc())
        return []

def get_template_image(platform):
    """Get template image based on platform"""
    template_images = {
        'Coursera': 'https://d3njjcbhbojbot.cloudfront.net/web/images/icons/coursera.svg',
        'Udemy': 'https://www.udemy.com/staticx/udemy/images/v7/logo-udemy.svg',
        'Analytics Vidhya': 'https://www.analyticsvidhya.com/wp-content/uploads/2015/12/av-logo.png',
        'edX': 'https://www.edx.org/sites/default/files/theme/edx-logo-header.png',
        'Khan Academy': 'https://cdn.kastatic.org/images/khan-logo-dark-background-2.png',
        'MIT OpenCourseWare': 'https://ocw.mit.edu/images/logo.png',
        'freeCodeCamp': 'https://www.freecodecamp.org/news/content/images/2020/10/fcc_primary.svg',
        'Harvard Online Learning': 'https://online-learning.harvard.edu/sites/default/files/styles/social_share/public/2019-11/HarvardX_Logo_Black.png'
    }
    return template_images.get(platform, 'https://via.placeholder.com/300x180?text=Course+Image')

def gradio_search(query):
    """Gradio search function with error handling"""
    try:
        if model is None:
            return """
            <div style="text-align: center; padding: 40px; color: white;">
                <h3>Error: Search model not initialized</h3>
                <p>Please try again in a few moments or contact support if the issue persists.</p>
            </div>
            """
            
        if not query or not query.strip():
            return """
            <div style="text-align: center; padding: 20px; color: white;">
                <h3>Please enter a search term</h3>
                <p>Try searching for topics like: Data Science, Python, AI, Machine Learning, etc.</p>
            </div>
            """
        
        # Get fresh data
        df = get_all_courses()
        if df.empty:
            return """
            <div style="text-align: center; padding: 40px; color: white;">
                <h3>No courses available</h3>
                <p>Please try again in a few moments.</p>
            </div>
            """
        
        result_list = search_courses(query, df)
        
        if result_list:
            html_output = """
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px;">
            """
            for item in result_list:
                try:
                    relevance_color = "#4CAF50" if item['score'] > 0.7 else "#FFC107" if item['score'] > 0.4 else "#FF5252"
                    
                    # Get template image if no image is available
                    image_url = item['image_url'] if item['image_url'] else get_template_image(item['platform'])
                    
                    # Add platform-specific styling
                    platform_styles = {
                        'Coursera': 'background: linear-gradient(45deg, #0056D2, #0077B5);',
                        'Udemy': 'background: linear-gradient(45deg, #A435F0, #8710D8);',
                        'Analytics Vidhya': 'background: linear-gradient(45deg, #FF6B6B, #FF8E8E);',
                        'edX': 'background: linear-gradient(45deg, #2A73FF, #1A63FF);',
                        'Khan Academy': 'background: linear-gradient(45deg, #14BF96, #0D9B7A);',
                        'MIT OpenCourseWare': 'background: linear-gradient(45deg, #8A8A8A, #666666);',
                        'freeCodeCamp': 'background: linear-gradient(45deg, #0A0A23, #1B1B32);',
                        'Harvard Online Learning': 'background: linear-gradient(45deg, #A51C30, #7D1414);'
                    }
                    
                    platform_style = platform_styles.get(item['platform'], 'background-color: #1e1e1e;')
                    
                    html_output += f"""
                    <div style="{platform_style} border-radius: 12px; width: 300px; padding: 16px; color: white; 
                         box-shadow: 0 4px 12px rgba(0,0,0,0.4); transition: all 0.3s ease;" 
                         onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 24px rgba(0,0,0,0.5)'" 
                         onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.4)'">
                        <div style="position: relative; width: 100%; height: 180px; overflow: hidden; border-radius: 8px; background-color: #2d2d2d;">
                            <img src="{image_url}" alt="{item['title']}" 
                                 style="width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease;"
                                 onerror="this.src='{get_template_image(item['platform'])}'"/>
                            <div style="position: absolute; top: 10px; right: 10px; background-color: rgba(0,0,0,0.7); 
                                      padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                                {item['platform']}
                            </div>
                        </div>
                        <h3 style="margin-top: 15px; font-size: 18px; line-height: 1.4; height: 50px; overflow: hidden; 
                                  text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                            {item['title']}
                        </h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0;">
                            <span style="background-color: rgba(255,255,255,0.1); padding: 5px 10px; border-radius: 15px; font-size: 14px;">
                                {item['platform']}
                            </span>
                            <span style="color: {relevance_color}; font-weight: bold; background-color: rgba(0,0,0,0.2); 
                                      padding: 5px 10px; border-radius: 15px;">
                                {item['score']:.2f}
                            </span>
                        </div>
                        <a href="{item['course_link']}" target="_blank" 
                           style="display: block; text-align: center; background-color: rgba(255,255,255,0.1); color: white; 
                                  padding: 12px; border-radius: 8px; text-decoration: none; margin-top: 15px;
                                  transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.2);"
                           onmouseover="this.style.backgroundColor='rgba(255,255,255,0.2)'; this.style.transform='translateY(-2px)'"
                           onmouseout="this.style.backgroundColor='rgba(255,255,255,0.1)'; this.style.transform='translateY(0)'">
                            Visit Course
                        </a>
                    </div>
                    """
                except Exception as e:
                    logger.error(f"Error rendering course card: {e}")
                    continue
                    
            html_output += "</div>"
            return html_output
        else:
            return """
            <div style="text-align: center; padding: 40px; color: white;">
                <h3>No courses found</h3>
                <p>Try different search terms or check back later for more courses.</p>
                <p>Suggested searches: Python, Data Science, Machine Learning, Web Development</p>
            </div>
            """
    except Exception as e:
        logger.error(f"Error in gradio_search: {e}")
        logger.error(traceback.format_exc())
        return """
        <div style="text-align: center; padding: 40px; color: white;">
            <h3>An error occurred</h3>
            <p>Please try again in a few moments.</p>
        </div>
        """

# Custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.gradio-interface {
    background-color: #121212 !important;
}
input[type="text"] {
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
    border: 2px solid #2d2d2d !important;
    background-color: #1e1e1e !important;
    color: white !important;
}
input[type="text"]:focus {
    border-color: #007bff !important;
    box-shadow: 0 0 0 2px rgba(0,123,255,0.25) !important;
}
.gradio-button {
    background-color: #007bff !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.gradio-button:hover {
    background-color: #0056b3 !important;
    transform: translateY(-1px) !important;
}
"""

iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(
        label="Search for a free course",
        placeholder="e.g., Data Science, Python, AI, Machine Learning",
        lines=1
    ),
    outputs=gr.HTML(label="Top Matching Courses"),
    title="🎓 Free Course Finder",
    description="""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2>Find Free Courses from Top Platforms</h2>
        <p>Search across Coursera, Udemy, and Analytics Vidhya for the best free courses.</p>
    </div>
    """,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Segoe UI", "system-ui", "sans-serif"]
    ),
    css=custom_css,
    allow_flagging="never"
)

if __name__ == "__main__":
    try:
        iface.launch(share=True)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        logger.error(traceback.format_exc())
