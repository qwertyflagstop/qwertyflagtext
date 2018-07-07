import numpy as np
import requests

class PushShift(object):

    def __init__(self, subreddit):
        super(PushShift, self).__init__()
        self.api_base = 'https://api.pushshift.io/reddit/search/submission/'
        self.sub = subreddit
        self.fp = 'trainTexts/{}.txt'.format(subreddit)
        self.posts_saved = 0
        self.ids = set()

    def get_posts_from(self,start,end):
        url_string = '{}?subreddit={}&after={}d&before={}d'.format(self.api_base,self.sub,start,end)
        url_string += '&size=500'
        url_string += '&num_comments=>0'
        r = requests.get(url_string)
        titles = set()
        for x in r.json()['data']:
                titles.add(x['title'])
        titles_txt = '\n'.join(titles)
        with open(self.fp,'a') as fp:
            fp.write(titles_txt)
        self.posts_saved += len(titles)

if __name__ == '__main__':
    sub = 'tifu'
    ps = PushShift(sub)
    for i in np.arange(365*10):
        ps.get_posts_from(i+1, i)
        print('Archived {} posts from {} going back {} days'.format(ps.posts_saved,sub, i))