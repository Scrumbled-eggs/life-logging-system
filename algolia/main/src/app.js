import { autocomplete } from '@algolia/autocomplete-js';
import instantsearch from 'instantsearch.js';
import { connectSearchBox } from 'instantsearch.js/es/connectors';
import historyRouter from 'instantsearch.js/es/lib/routers/history';
import {
  configure,
  refinementList,
  hits,
  pagination,
  sortBy,
} from 'instantsearch.js/es/widgets';

const searchClient = algoliasearch(
  '0L0TPDZHFM',
  '1a42927a7a1ffc3661c466e3a7acda87'
);

const INSTANT_SEARCH_INDEX_NAME = 'milestone1'
const instantSearchRouter = historyRouter()

const search = instantsearch({
  indexName: INSTANT_SEARCH_INDEX_NAME,
  searchClient,
  routing: instantSearchRouter
});

const virtualSearchBox = connectSearchBox(() => {})

search.addWidgets([
  virtualSearchBox({}),
  refinementList({
    container: '#filter-list',
    attribute: 'filelength(s)',
  }),
  refinementList({
    container: '#fps-list',
    attribute: 'fps',
    showMore: true,
    soryBy: 'count:desc',
  }),
  configure({
    hitsPerPage: 16
  }),
  hits({
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
    },
  }),
  configure({
    facets: ['*'],
    maxValuesPerFacet: 20,
  }),
  pagination({
    container: '#pagination',
    showFirst: true,
    showLast: true
  }),
]);

//TBD
sortBy({
  container: '#sort-by',
  items: [
    { label: 'Featured', value: 'instant_search' },
    { label: 'Price (asc)', value: 'instant_search_price_asc' },
    { label: 'Price (desc)', value: 'instant_search_price_desc' },
  ],
});


search.start();

// Set the InstantSearch index UI state from external events.
function setInstantSearchUiState(indexUiState) {
  search.setUiState(uiState => ({
    ...uiState,
    [INSTANT_SEARCH_INDEX_NAME]: {
      ...uiState[INSTANT_SEARCH_INDEX_NAME],
      // We reset the page when the search state changes.
      page: 1,
      ...indexUiState,
    },
  }))
}

// Return the InstantSearch index UI state.
function getInstantSearchUiState() {
  const uiState = instantSearchRouter.read()

  return (uiState && uiState[INSTANT_SEARCH_INDEX_NAME]) || {}
}

const searchPageState = getInstantSearchUiState()

autocomplete({
  container: '#autocomplete',
  placeholder: 'Search for products',
  detachedMediaQuery: 'none',
  initialState: {
    query: searchPageState.query || '',
  },
  onSubmit({ state }) {
    setInstantSearchUiState({ query: state.query })
  },
  onReset() {
    setInstantSearchUiState({ query: '' })
  },
  onStateChange({ prevState, state }) {
    if (prevState.query !== state.query) {
      setInstantSearchUiState({ query: state.query })
    }
  },
})
