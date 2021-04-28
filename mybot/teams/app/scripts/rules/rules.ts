export interface Rules {
    comments?: string
    urlPattern: string
    rules: Rule[]
}

export interface Rule {
    messageExactMatch?: string
    messagePattern?: string
    regexFlags?: string
    response: string
}

export interface RuleSettings {
    dateModified: Date
    apps: Rules[]
}