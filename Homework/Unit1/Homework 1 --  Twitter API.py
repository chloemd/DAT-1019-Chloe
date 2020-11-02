#!/usr/bin/env python
# coding: utf-8


# ## <font color=red>Homework header functions and variables (include in .py)</font>

# In[8]:


import requests
from requests_oauthlib import OAuth1
import pandas as pd
api_key = 'vbw5MI6thcAX6909ooSURWBbt'
api_secret = 'TiNGvc0MolCtZBJ03AxjRBkaR0OEXmaj0EVWHxGOdVMvzyZ93X'
access_token = '1081195307378098177-V0fO98sLUJWo4sv3IUbASbNcfyKIEG'
access_secret = 'Rfd5veGIr18sv30hUx4VGqph6LvjNrIdz4Yo8A2JNlfd1'

search_url = 'https://api.twitter.com/1.1/search/tweets.json'
followers_url = 'https://api.twitter.com/1.1/followers/list.json'
users_url = 'https://api.twitter.com/1.1/users/show.json'
friends_url = 'https://api.twitter.com/1.1/friends/list.json'

auth = OAuth1(api_key, api_secret, access_token, access_secret)


# ##### Function 1 (Required)

# **Name:** `find_user`
# 
# **Returns:** dictionary that represents user object returned by Twitter's API
# 
# **Arguments:**
#  - `screen_name`: str, required; Twitter handle to search for.  **Can include the @ symbol.  The function should check for this and remove it if necessary.**
#  - `keys`: list, optional; list that contains keys to return about user object.  If not specified, then function should return the entire user object.  **These only need to be outer keys.** If they are keys nested within another key, you don't have to account for this.
#  
# **To test:** We'll test your function in the following ways:
# 
#  - `find_user('@GA')`
#  - `find_user('GA')`
#  - `find_user('GA', keys=['name', 'screen_name', 'followers_count', 'friends_count'])`

# ### <font color=green>Function 1 Write-Up</font>
# According to <a href="https://help.twitter.com/en/managing-your-account/twitter-username-rules#:~:text=Your%20username%20cannot%20be%20longer,of%20underscores%2C%20as%20noted%20above.">Twitter username requirements</a>, valid Twitter names contain only **alphanumeric characters and underscores**. Therefore I can remove any '@' symbols in usernames before passing them to my query without causing unintended issues.

# In[9]:


def find_user(screen_name, keys=[]):
    """Returns a dictionary with user data. If no keys are specified, the entire user object is returned."""
    screen_name = screen_name.strip('@')
    results = requests.get((users_url+'?screen_name='+screen_name), auth=auth).json()
    if any(keys):
        # only iterating over keys contained in dict to prevent errors
        return {key:results[key] for key in keys if key in results.keys()}
    else:
        return results


# ##### Function 2 (Required)

# **Name:** `find_hashtag`
# 
# **Returns:** list of data objects that contain information about each tweet that matches the hashtag provided as input.
# 
# **Arguments:**
#  - `hashtag`: str, required; text to use as a hashtag search.  
#  - `count`: int, optional; number of results to return
#  - `search_type`: str, optional; type of results to return.  should accept 3 different values:
#    - `mixed`:   return mix of most recent and most popular results
#    - `recent`:  return most recent results
#    - `popular`: return most popular results
#    
# **Note:** User should **not** have to actually use the `#` character for the `hashtag` argument.  The function should check to see if it's there, and if not, add it in for them.
# 
# **To Test:**  We'll check your function in the following ways:
#  - `find_hashtag('DataScience')`
#  - `find_hashtag('#DataScience')`
#  - `find_hashtag('#DataScience', count=100)`, and double check the length of the `statuses` key to make sure it contains the right amount of results.  **Note:** Due to the version of the API we're using, the number of results returned will **not** necessarily match the value passed into the `count` parameter.  So if you specify 50 and it only returns 45, you are likely still doing it correctly.
#  - `find_hashtag('#DataScience', search_type='recent/mixed/popular')`

# ### <font color=green>Function 2 Write-Up</font>
# 

# In[10]:


def find_hashtag(hashtag, count=10, search_type='mixed'):
    """Returns a list of tweet data for tweets matching the hashtag."""
    # checking for and repairing bad values passed as args
    if search_type not in ['mixed', 'popular', 'recent']:
        search_type = 'mixed'
    try:
        count = str(int(count))
    except ValueError:
        count = '10'
    
    hashtag = hashtag.strip('#')
    results = requests.get((search_url+'?q=%23'+hashtag+'&count='+count+'&search_type='+search_type), auth=auth).json()
    return [tweet for tweet in results['statuses']]


# ##### Function 3 (Required)

# **Name:** `get_followers`
# 
# **Returns:** list of data objects for each of the users followers, returning values for the `name`, `followers_count`, `friends_count`, and `screen_name` key for each user.
# 
# **Arguments:** 
# 
#  - `screen_name`: str, required; Twitter handle to search for.  **Results should not depend on user inputting the @ symbol.**
#  - `keys`: list, required;  keys to return for each user.  default value: [`name`, `followers_count`, `friends_count`, `screen_name`]; if something else is listed, values for those keys should be returned
#  - `to_df`: bool, required; default value: False; if True, return results in a dataframe.  Every value provided in the `keys` argument should be its own column, with rows populated by the corresponding values for each one for every user.
#  
# **To Test:** We'll test your functions in the following ways:
# 
#  - `get_followers('@GA')`
#  - `get_followers('GA')`
#  - `get_followers('GA', keys=['name', 'followers_count'])`
#  - `get_followers('GA', keys=['name', 'followers_count'], to_df=True)`
#  - `get_followers('GA', to_df=True)`

# ### <font color=green>Function 3 Write-Up</font>

# In[11]:


def get_followers(screen_name, keys=['name', 'followers_count', 'friends_count', 'screen_name'], to_df=False):
    """Returns followers of a specified user. Can specify columns to return, but defaults are available as well."""
    screen_name = screen_name.strip('@')
    if not any(keys):
        keys = ['name', 'followers_count', 'friends_count', 'screen_name']
    try:
        # prevent the function from throwing an error if something goes wrong with the GET request
        results = requests.get((followers_url+'?screen_name='+screen_name), auth=auth).json()['users']
    except KeyError:
        return 'API limits exhausted or invalid screen name. Please try again in 15 minutes.'
    data = []
    for result in results:
        data.append({key:result[key] for key in keys if key in result.keys()})
    if to_df:        
        return pd.DataFrame(data)
    else:
        return data



# ##### Function 4 (Optional)

# **Name:** `friends_of_friends`
# 
# **Returns:** list of data objects for each user that two Twitter users have in common
# 
# **Arguments:**
# 
#  - `names`: list, required; list of two Twitter users to compare friends list with
#  - `keys`: list, optional; list of keys to return for information about each user.  Default value should be to return the entire data object.
#  - `to_df`: bool, required; default value: False; if True, returns results in a dataframe.
#  
# **To Test:** We'll test your function in the following ways:
# 
#  - `friends_of_friends(['Beyonce', 'MariahCarey'])`
#  - `friends_of_friends(['@Beyonce', '@MariahCarey'], to_df=True)`
#  - `friends_of_friends(['Beyonce', 'MariahCarey'], keys=['id', 'name'])`
#  - `friends_of_friends(['Beyonce', 'MariahCarey'], keys=['id', 'name'], to_df=True)`
#  
# Each of these should return 3 results. (Assuming they haven't followed the same people since this was last written).  
# 
# **Hint:** The `id` key is the unique identifier for someone, so if you want to check if two people are the same this is the best way to do it.

# ### <font color=green>Function 4 Write-Up</font>

# In[46]:


def friends_of_friends(names, keys=[], to_df=False):
    """Returns friends that two users have in common"""
    person1 = names[0].strip('@')
    person2 = names[1].strip('@')
    common_friends = []
    try:
        # prevent the function from throwing an error if something goes wrong with the GET request
        person1_friends = requests.get((friends_url+'?screen_name='+person1+'&count=200'), auth=auth).json()['users']
        person2_friends = requests.get((friends_url+'?screen_name='+person2+'&count=200'), auth=auth).json()['users']
    except KeyError:
        return 'API limits exhausted or invalid screen name. Please try again in 15 minutes.'
    for p1 in person1_friends:
        for p2 in person2_friends:
            if p1['id'] == p2['id']:
                if not any(keys):
                    common_friends.append(p1)
                else:
                    common_friends.append({key:p1[key] for key in keys if key in p1.keys()})
    if to_df:
        return pd.DataFrame(common_friends)
    else:
        return common_friends
    
friends_of_friends(['Beyonce', 'MariahCarey'], keys=['id', 'name'], to_df=True)


#  ##### Function 5 (Optional)

# Rewrite the `friends_of_friends` function, except this time include an argument called `full_search`, which accepts a boolean value.  If set to `True`, use cursoring to cycle through the complete set of users for the users provided.  
# 
# The twitter API only returns a subset of users in your results to save bandwidth, so you have to cycle through multiple result sets to get all of the values.
# 
# You can read more about how this works here:  https://developer.twitter.com/en/docs/basics/cursoring
# 
# Basically you have to do a `while` loop to continually make a new request using the values stored in the `next_cursor` key as part of your next query string until there's nothing left to search.
# 
# **Note:** We're using the free API, so we're operating under some limitations.  One of them being that you can only make 15 API calls in a 15 minute span to this portion of the API.  You can also only return up to 200 results per cursor, so this means you won't be able to completely search for everyone even if you set this up correctly.
# 
# That's fine, just do what you can under the circumstances.
# 
# **To Test:** To test your function, we'll run the following function calls:
# 
#  - `friends_of_friends(['ezraklein', 'tylercowen'])` -- should return 4 results if you do an API call that returns 200 results
#  - `friends_of_friends(['ezraklein', 'tylercowen'], full_search=True)` -- should return 54 results if you do an API call that returns 200 results
#  
# **Hint:** Chances are you will exhaust your API limits quite easily in this function depending on who you search for.  Depending on how you have things set up, this could cause error messages to arise when things are otherwise fine.  Remember in class 3 when we were getting those weird dictionaries back because our limits were used up?  We won't hold you accountable for handling this inside your function, although it could make some things easier for your own testing.
#        
# Good luck!

# ### <font color=green>Function 5 Write-Up</font>

# In[62]:


def friends_of_friends_helper(list1, list2, keys):
    """Helper function for friends_of_friends created to avoid repeating code with the cursoring feature. Returns a list."""
    common_friends = []
    for p1 in list1:
        for p2 in list2:
            if p1['id'] == p2['id']:
                if not any(keys):
                    common_friends.append(p1)
                else:
                    common_friends.append({key:p1[key] for key in keys if key in p1.keys()})
    return common_friends

def friends_of_friends(names, keys=[], to_df=False, full_search=False):
    person1 = names[0].strip('@')
    person2 = names[1].strip('@')
    common_friends = []
    try:
        # prevent the function from throwing an error if something goes wrong with the GET request
        person1_request = requests.get((friends_url+'?screen_name='+person1+'&count=200'), auth=auth).json()
        person1_friends = person1_request['users']
        p1_cursor = person1_request['next_cursor_str']
        
        person2_request = requests.get((friends_url+'?screen_name='+person2+'&count=200'), auth=auth).json()
        person2_friends = person2_request['users']
        p2_cursor = person2_request['next_cursor_str']
    except KeyError:
        return 'API limits exhausted or invalid screen name. Please try again in 15 minutes.'
    
    
    if full_search:
        # full search will require cursoring to get all of the results. otherwise just use the initial friends lists generated above
        while int(p1_cursor) > 0:
            try:
                person1_request = requests.get((friends_url+'?screen_name='+person1+'&count=200'+'&cursor='+p1_cursor), auth=auth).json()
                person1_friends += (user for user in person1_request['users'])
                p1_cursor = person1_request['next_cursor_str']
                # print(f"Next P1 Cursor: {p1_cursor}")
            except KeyError:
                # breaking here (instead of returning an error) because at least some results can be generated if this part of the code is being evaluated
                pass
            
        while int(p2_cursor) > 0:
            try:
                person2_request = requests.get((friends_url+'?screen_name='+person2+'&count=200'+'&cursor='+p2_cursor), auth=auth).json()
                person2_friends += (user for user in person2_request['users'])
                p2_cursor = person2_request['next_cursor_str']
                # print(f"Next P2 Cursor: {p2_cursor}")
            except KeyError:
                # breaking here (instead of returning an error) because at least some results can be generated if this part of the code is being evaluated
                pass
        
            
    common_friends = friends_of_friends_helper(person1_friends, person2_friends, keys)
                    
    # checking whether to return result as DataFrame type or list of dictionaries:
    if to_df:
        return pd.DataFrame(common_friends)
    else:
        return common_friends

friends_of_friends(['ezraklein', 'tylercowen'], keys=['id', 'name'], to_df=True, full_search=True)

