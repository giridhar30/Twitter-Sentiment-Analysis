
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import sentiment_mod as s

import json

ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        tweet = all_data["text"]
        
        sent_val, conf = s.sentiment(tweet)

        print(tweet)
        print(sent_val, conf)

        if conf*100 >= 80:
        	op = open('twitter-out.txt','a')
        	op.write(sent_val)
        	op.write('\n')
        	op.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
