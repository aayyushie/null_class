def search_papers(df, query):
    # Simple search: filter by title, author, abstract
    mask = df['title'].str.contains(query, case=False) | \
           df['authors'].str.contains(query, case=False) | \
           df['abstract'].str.contains(query, case=False)
    return df[mask].head(10)