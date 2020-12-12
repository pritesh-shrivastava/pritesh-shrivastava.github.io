Personal website & blog hosted at [pritesh-shrivastava.github.io](https://pritesh-shrivastava.github.io/)


To test the site locally, run
```
bundle exec jekyll serve
```

To preview future dated posts, use
```
bundle exec jekyll serve --future
```

## Next steps

#### Site improvements
- Add link to Facebook profile ?
- Add Resume & other Project details with Git repos
- Import stats from Goodreads profile
- Add categories / collections for post links from mm ?
- Year to year-month archives ?
- Check PageSpeed suggestions from Google Analytics if score < 90% for a page
- Add Disqus activity to Google Analytics as goals or events - responses & comments
- Wanna clean GA / comments according to MM ? They are both working currently. Changing it might lead to loss of old comments / responses or page views!
- Connection to the site is not secure due to images ?!


#### Blog post ideas
- Kaggle competitions
- Add book reviews & MOOC reviews
- Prog Lang course - Parts B & C
- More Haskell - Monoid, purity
- HtDP exercise on dictionary - Ch 2 Ex199-204 + Word combinations
- College / branch vs career
- Insertion sort from Racket code in HtDP - helper functions that recur
- Classification & k-means clustering of prog lang text - Kaggle Dataset
- Image processing (ala CS50 assignment) in R / Julia
- Summary of Google Analytics data for 2020 in R
- What I learnt from blogging
- Matrix operation via recursion on Jupyter in Scheme


#### Promotion
- Cross publish articles to Medium / HN
    - Can import artices to these sites with their canonical links. Simple copy pasting content without providing canonical link can penalize SEO
    - Code & markdown chunks are not rendered properly on Medium. Need text & atleast 1 image per article
        - Can try importing Github gists to Medium
        - Use Python package [jupyter_to_medium](https://pypi.org/project/jupyter-to-medium/)
- Reddit / Slack / Twitter / LinkedIn / FB groups & LinkedIn groups
- Publish R blogs on R-weekly by link or R-blogger with a separate RSS feed for posts with tag R
    https://www.r-bloggers.com/add-your-blog/
    Solutions - 
    [1] https://gist.github.com/hunleyd/95d2081d339bddd45dd4189275892a13
    [2] https://github.com/oxinabox/oxinabox.github.io/blob/master/rssfeed_julia.xml#L13

