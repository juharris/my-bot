import { PaletteType } from '@material-ui/core'
import { browser } from 'webextension-polyfill-ts'

export type ThemePreferenceType = PaletteType | 'device'

export interface UserSettings {
	themePreference: ThemePreferenceType
}

export async function setupUserSettings(requiredKeys: (keyof (UserSettings))[]): Promise<UserSettings> {
	const keys = {
		themePreference: undefined,
	}

	const results = await browser.storage.local.get(keys)
	let { themePreference } = results
	if (requiredKeys.indexOf('themePreference') > - 1 && themePreference === undefined) {
		const r = await browser.storage.sync.get(['themePreference'])
		themePreference = r.themePreference
		if (themePreference !== undefined) {
			browser.storage.local.set({ themePreference })
		} else {
			themePreference = 'device'
		}
	}

	const result: any = { themePreference }
	return result
}
