# Online Recommender System Playground

## TODO

### Recommender System

- Fix weight updates. It's kind of wonky, especially with thumbs down. The algorithm is... iffy. Maybe improve with some theory.
    - Probably I can just use serialized scikit-learn logistic regression and do partial_fits
- Maybe apply sigmoid to scores, to move this closer to a logistic regression?
- Check out some alternative vectorization? Maybe vectorization of plain text with full synposis is... wrong. Maybe some recsys standard will serve us better (like, some matrix factorization?)
- Check out alternative dataset. Ideas:
    - Retina Latina
    - Look into letterboxd API? The "themes" and genres there are really cool
- Build support for dynamic batch sizes (don't update user vector every x times)


### Front end

- It can be better, but it's fine for now...

### Technical debt

- Split out stuff into files
- Add typing for movies, etc (Pydantic models)
- Move tests into tests
- Add a makefile for the different processes?
- Fill in this README with actual setup instructions