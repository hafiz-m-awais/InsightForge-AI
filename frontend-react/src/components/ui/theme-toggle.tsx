import { Moon, Sun, Monitor } from 'lucide-react'
import { useTheme } from '@/components/theme-provider'

export function ThemeToggle({ collapsed = false }: { collapsed?: boolean }) {
  const { theme, setTheme } = useTheme()

  const nextTheme = () => {
    if (theme === 'light') setTheme('dark')
    else if (theme === 'dark') setTheme('system')
    else setTheme('light')
  }

  const getIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="w-4 h-4" />
      case 'dark':
        return <Moon className="w-4 h-4" />
      default:
        return <Monitor className="w-4 h-4" />
    }
  }

  const getLabel = () => {
    switch (theme) {
      case 'light':
        return 'Light'
      case 'dark':
        return 'Dark'
      default:
        return 'System'
    }
  }

  return (
    <button
      onClick={nextTheme}
      className={
        collapsed
          ? 'flex items-center justify-center w-full py-2 text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors'
          : 'flex items-center gap-2 w-full px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors'
      }
      title={`Theme: ${getLabel()}. Click to cycle.`}
    >
      {getIcon()}
      {!collapsed && <span className="flex-1 text-left">{getLabel()}</span>}
    </button>
  )
}