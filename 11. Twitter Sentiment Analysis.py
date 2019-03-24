from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s


#consumer key, consumer secret, access token, access secret.
ckey="9obSPVDuBDEwR2w9rwmX2y3M8"
csecret="6Ed8GW0OMfocJgIGzX5ofc7AYg7JLXPP14vlFB3aXedV4RiKZY"
atoken="1107023002565148672-SSs5JKSzNHpfeEZNAEXwjI3OncTgpd"
asecret="7oGWow4RZmbKxshV1lWI5Tl4lps8ipQnyQvev0HLzBhE4"

class listener(StreamListener):

    def on_data(self, data):
        try:
            
            all_data = json.loads(data)
            
            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence*100
                  )

            if confidence*100 >=80:
                output = open("twitter-out.txt","a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()
                
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
