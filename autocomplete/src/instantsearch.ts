import instantsearch from 'instantsearch.js';
import algoliasearch from 'algoliasearch/lite';
import { connectSearchBox } from 'instantsearch.js/es/connectors';
import historyRouter from 'instantsearch.js/es/lib/routers/history';
import {
  configure,
  hierarchicalMenu,
  hits,
  pagination,
  panel,
} from 'instantsearch.js/es/widgets';

import { debounce } from '../src/debounce';

export const INSTANT_SEARCH_INDEX_NAME = 'milestone1';
export const INSTANT_SEARCH_HIERARCHICAL_ATTRIBUTE =
  'hierarchicalCategories.lvl0';

const instantSearchRouter = historyRouter();
const searchClient = algoliasearch(
  '0L0TPDZHFM',
  '1a42927a7a1ffc3661c466e3a7acda87'
);

export const search = instantsearch({
  searchClient,
  indexName: INSTANT_SEARCH_INDEX_NAME,
  routing: instantSearchRouter,
});
const virtualSearchBox = connectSearchBox(() => {});
const hierarchicalMenuWithHeader = panel({
  templates: { header: 'Categories' },
})(hierarchicalMenu);

search.addWidgets([
  configure({
    attributesToSnippet: ['name:7', 'description:15'],
    snippetEllipsisText: '…',
  }),
  // Mount a virtual search box to manipulate InstantSearch's `query` UI
  // state parameter.
  virtualSearchBox(),
  hierarchicalMenuWithHeader({
    container: '#categories',
    attributes: [
      INSTANT_SEARCH_HIERARCHICAL_ATTRIBUTE,
      'hierarchicalCategories.lvl1',
    ],
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
  pagination({
    container: '#pagination',
    padding: 2,
    showFirst: false,
    showLast: false,
  }),
]);

// Set the InstantSearch index UI state from external events.
export function setInstantSearchUiState(indexUiState) {
  search.setUiState((uiState) => ({
    ...uiState,
    [INSTANT_SEARCH_INDEX_NAME]: {
      ...uiState[INSTANT_SEARCH_INDEX_NAME],
      // We reset the page when the search state changes.
      page: 1,
      ...indexUiState,
    },
  }));
}

export const debouncedSetInstantSearchUiState = debounce(
  setInstantSearchUiState,
  500
);

// Get the current category from InstantSearch.
export function getInstantSearchCurrentCategory() {
  const indexUiState = search.getUiState()[INSTANT_SEARCH_INDEX_NAME];
  const hierarchicalMenuUiState = indexUiState && indexUiState.hierarchicalMenu;
  const currentCategories =
    hierarchicalMenuUiState &&
    hierarchicalMenuUiState[INSTANT_SEARCH_HIERARCHICAL_ATTRIBUTE];

  return currentCategories && currentCategories[0];
}

// Build URLs that InstantSearch understands.
export function getInstantSearchUrl(indexUiState) {
  return search.createURL({ [INSTANT_SEARCH_INDEX_NAME]: indexUiState });
}

// Return the InstantSearch index UI state.
export function getInstantSearchUiState() {
  const uiState = instantSearchRouter.read();

  return (uiState && uiState[INSTANT_SEARCH_INDEX_NAME]) || {};
}
