import { t } from '../i18n.js';

export function formatIngestionError(error) {
  switch (error?.code) {
    case 'invalid_repo_id':
      return t('err.badRepoId');
    case 'config_fetch_failed':
      return `${t('err.configFetch')}${error.details?.message || ''}`;
    case 'provenance_fetch_failed':
      return `${t('err.provenanceFetch')}${error.details?.message || ''}`;
    case 'safetensors_unavailable':
      return t('err.noSafetensors');
    case 'invalid_safetensors_header':
      return t('err.badSafetensors', { file: error.details?.file || '—' });
    default:
      return error?.message || String(error);
  }
}
