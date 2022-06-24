import instantsearch from 'instantsearch.js';
import algoliasearch from 'algoliasearch/lite';
import { autocomplete } from '@algolia/autocomplete-js';
import { connectSearchBox } from 'instantsearch.js/es/connectors';
import historyRouter from 'instantsearch.js/es/lib/routers/history';
import { createQuerySuggestionsPlugin } from '@algolia/autocomplete-plugin-query-suggestions';
import { createLocalStorageRecentSearchesPlugin } from '@algolia/autocomplete-plugin-recent-searches';
import { h, Fragment } from 'preact';

import {
  configure,
  refinementList,
  hits,
  pagination,
  sortBy,
} from 'instantsearch.js/es/widgets';

import '@algolia/autocomplete-theme-classic'

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
  refinementList({
    container: '#filter-list',
    attribute: 'filelength(s)',
  }),
  refinementList({
    container: '#fps-list',
    attribute: 'fps',
    showMore: true
  }),
  configure({
    hitsPerPage: 16
  }),
  virtualSearchBox({}),
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
    { label: 'Featured', value: 'milestone1' }
    ],
});


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

function debounce(fn, time) {
  let timerId = undefined

  return function(...args) {
    if (timerId) {
      clearTimeout(timerId)
    }

    timerId = setTimeout(() => fn(...args), time)
  }
}

const debouncedSetInstantSearchUiState = debounce(setInstantSearchUiState, 500)


const searchPageState = getInstantSearchUiState()

// Build URLs that InstantSearch understands.
function getInstantSearchUrl(indexUiState) {
  return search.createURL({ [INSTANT_SEARCH_INDEX_NAME]: indexUiState });
}

// Return the InstantSearch index UI state.
function getInstantSearchUiState() {
  const uiState = instantSearchRouter.read()

  return (uiState && uiState[INSTANT_SEARCH_INDEX_NAME]) || {}
}

search.start();


// Detect when an event is modified with a special key to let the browser
// trigger its default behavior.
function isModifierEvent(event) {
  const isMiddleClick = event.button === 1;

  return (
    isMiddleClick ||
    event.altKey ||
    event.ctrlKey ||
    event.metaKey ||
    event.shiftKey
  );
}

function onSelect({ setIsOpen, setQuery, event, query }) {
  // You want to trigger the default browser behavior if the event is modified.
  if (isModifierEvent(event)) {
    return;
  }

  setQuery(query);
  setIsOpen(false);
  setInstantSearchUiState({ query });
}

function getItemUrl({ query }) {
  return getInstantSearchUrl({ query });
}

function createItemWrapperTemplate({ children, query, html }) {
  const uiState = { query };

  return html`<a
    class="aa-ItemLink"
    href="${getInstantSearchUrl(uiState)}"
    onClick="${(event) => {
      if (!isModifierEvent(event)) {
        // Bypass the original link behavior if there's no event modifier
        // to set the InstantSearch UI state without reloading the page.
        event.preventDefault();
      }
    }}"
  >
    ${children}
  </a>`;
}

const recentSearchesPlugin = createLocalStorageRecentSearchesPlugin({
  key: 'instantsearch',
  limit: 5,
  transformSource({ source }) {
    return {
      ...source,
      getItemUrl({ item }) {
        return getItemUrl({
          query: item.label,
        });
      },
      onSelect({ setIsOpen, setQuery, item, event }) {
        onSelect({
          setQuery,
          setIsOpen,
          event,
          query: item.label,
        });
      },
      // Update the default `item` template to wrap it with a link
      // and plug it to the InstantSearch router.
      templates: {
        ...source.templates,
        item(params) {
          const { children } = source.templates.item(params).props;

          return createItemWrapperTemplate({
            query: params.item.label,
            children,
            html: params.html,
          });
        },
      },
    };
  },
});

const querySuggestionsPlugin = createQuerySuggestionsPlugin({
  searchClient,
  indexName: 'milestone1_query_suggestions3',
  getSearchParams({ state }) {
    return { hitsPerPage: state.query ? 5 : 10};
    // This creates a shared `hitsPerPage` value once the duplicates
    // between recent searches and Query Suggestions are removed.
    // return recentSearchesPlugin.data.getAlgoliaSearchParams({
    //  hitsPerPage: 6,
    // });
  },
  transformSource({ source }) {
    return {
      ...source,
      sourceId: 'querySuggestionsPlugin',
      getItemUrl({ item }) {
        return getItemUrl({
          query: item.query,
        });
      },
      onSelect({ setIsOpen, setQuery, event, item }) {
        onSelect({
          setQuery,
          setIsOpen,
          event,
          query: item.query,
        });
      },
      getItems(params) {
        return source.getItems(params);
      },
      templates: {
        ...source.templates,
        item(params) {
          const { children } = source.templates.item(params).props;

          return createItemWrapperTemplate({
            query: params.item.label,
            children,
            html: params.html,
          });
        },
      },
    };
  },
});


autocomplete({
  // You want recent searches to appear with an empty query.
  openOnFocus: true,
  // Add the recent searches and Query Suggestions plugins.
  plugins: [recentSearchesPlugin, querySuggestionsPlugin],
  container: '#autocomplete',
  placeholder: 'Search for video..',
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
