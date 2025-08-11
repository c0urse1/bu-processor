# Git Repository Reparatur - Anleitung

## Problem
- Git Verbindung zu GitHub ist kaputt
- Source Control zeigt "Master" statt "Main"
- Push funktioniert nicht mehr

## Lösung 1: Automatische Reparatur (empfohlen)

### Option A: Batch Skript
```cmd
# Einfach doppelklicken oder in CMD ausführen:
git_reconnect_fix.bat
```

### Option B: PowerShell Skript
```powershell
# Rechtsklick → "Mit PowerShell ausführen" oder:
powershell -ExecutionPolicy Bypass -File git_reconnect_fix.ps1
```

## Lösung 2: Manuelle Schritte

### 1. Aktuelle Remote-Verbindung prüfen
```cmd
git remote -v
```

### 2. Kaputte Remote-Verbindung entfernen
```cmd
git remote remove origin
```

### 3. Neue Remote-Verbindung hinzufügen
```cmd
# Ersetze URL mit deiner echten GitHub Repository URL
git remote add origin https://github.com/DEIN_USERNAME/DEIN_REPO.git
```

### 4. Alle Branches vom Remote laden
```cmd
git fetch origin
```

### 5. Auf Main Branch wechseln
```cmd
# Wenn main Branch existiert:
git checkout -b main origin/main

# Oder wenn nur master existiert:
git checkout -b main origin/master
git push origin main
```

### 6. Upstream setzen
```cmd
git branch --set-upstream-to=origin/main main
```

### 7. Status prüfen
```cmd
git branch -a
git status
```

### 8. Code pushen
```cmd
git add .
git commit -m "Reconnect to GitHub main branch"
git push
```

## Häufige GitHub Repository URLs

- **HTTPS**: `https://github.com/USERNAME/REPOSITORY.git`
- **SSH**: `git@github.com:USERNAME/REPOSITORY.git`

## Nach der Reparatur

1. ✅ Source Control sollte "main" anzeigen
2. ✅ `git status` sollte "On branch main" zeigen
3. ✅ `git push` sollte wieder funktionieren

## Troubleshooting

### Fehler: "Permission denied"
- GitHub Token/Passwort abgelaufen
- Lösung: In Windows Credential Manager GitHub Credentials aktualisieren

### Fehler: "Repository not found"
- Repository URL falsch
- Repository ist privat und du hast keinen Zugriff

### Fehler: "Branch main not found"
- Repository nutzt noch master als Standard
- Lösung: Skript wechselt automatisch oder erstellt main Branch

## Backup vor Änderungen
```cmd
# Lokale Änderungen sichern
git stash
git stash list

# Nach Reparatur wiederherstellen
git stash pop
```
