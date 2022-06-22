const { algoliasearch , instantsearch  } = window;
const searchClient = algoliasearch('0L0TPDZHFM', '1a42927a7a1ffc3661c466e3a7acda87');
const INSTANT_SEARCH_INDEX_NAME = 'milestone1';
const instantSearchRouter = historyRouter();
const search = instantsearch({
    indexName: INSTANT_SEARCH_INDEX_NAME,
    searchClient,
    routing: instantSearchRouter
});
const virtualSearchBox = connectSearchBox(()=>{});
search.addWidgets([
    virtualSearchBox({}),
    instantsearch.widgets.refinementList({
        container: '#filter-list',
        attribute: 'filelength(s)'
    }),
    instantsearch.widgets.refinementList({
        container: '#fps-list',
        attribute: 'fps',
        showMore: true,
        soryBy: 'count:desc'
    }),
    instantsearch.widgets.configure({
        hitsPerPage: 16
    }),
    instantsearch.widgets.hits({
        container: '#hits',
        templates: {
            item: `
<div>
      <div class="hit-face">
        {{#helpers.highlight}}{ "attribute": "face" }{{/helpers.highlight}}
      </div>
      <div class="hit-action">
        {{#helpers.highlight}}{ "attribute": "action" }{{/helpers.highlight}}
      </div>
      <div class="hit-file">
      filename:
      {{#helpers.highlight}}{ "attribute": "filename" }{{/helpers.highlight}}
      </div>
      <div class="hit-frame">frame:{{frame}}</div>
      <div class="hit-timestamp">timestamp(s):{{timestamp(s)}}</div>
      <div class="hit-filelen">filelength(s):{{filelength(s)}}</div>
      <div class="hit-fps">fps:{{fps}}</div>
    </div>
`,
            empty: 'No result for <q>{{ query }}</q>'
        }
    }),
    instantsearch.widgets.configure({
        facets: [
            '*'
        ],
        maxValuesPerFacet: 20
    }),
    instantsearch.widgets.pagination({
        container: '#pagination',
        showFirst: true,
        showLast: true
    }), 
]);
//TBD
instantsearch.widgets.sortBy({
    container: '#sort-by',
    items: [
        {
            label: 'Featured',
            value: 'instant_search'
        },
        {
            label: 'Price (asc)',
            value: 'instant_search_price_asc'
        },
        {
            label: 'Price (desc)',
            value: 'instant_search_price_desc'
        }, 
    ]
});
search.start();
// Set the InstantSearch index UI state from external events.
function setInstantSearchUiState(indexUiState) {
    search.setUiState((uiState)=>({
            ...uiState,
            [INSTANT_SEARCH_INDEX_NAME]: {
                ...uiState[INSTANT_SEARCH_INDEX_NAME],
                // We reset the page when the search state changes.
                page: 1,
                ...indexUiState
            }
        })
    );
}
// Return the InstantSearch index UI state.
function getInstantSearchUiState() {
    const uiState = instantSearchRouter.read();
    return uiState && uiState[INSTANT_SEARCH_INDEX_NAME] || {};
}
const searchPageState = getInstantSearchUiState();
autocomplete({
    container: '#autocomplete',
    placeholder: 'Search for products',
    detachedMediaQuery: 'none',
    initialState: {
        query: searchPageState.query || ''
    },
    onSubmit ({ state  }) {
        setInstantSearchUiState({
            query: state.query
        });
    },
    onReset () {
        setInstantSearchUiState({
            query: ''
        });
    },
    onStateChange ({ prevState , state  }) {
        if (prevState.query !== state.query) setInstantSearchUiState({
            query: state.query
        });
    }
});

//# sourceMappingURL=index.816e7b21.js.map
