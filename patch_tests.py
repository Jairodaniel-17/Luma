import os
import glob

for root, _, files in os.walk('tests'):
    for file in files:
        if file.endswith('.rs'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            
            # replace .send() with .bearer_auth("test").send()
            # but only for client requests, not other .send() like oneshot channels
            # client sends are usually .send().await
            content = content.replace(".send()\n", '.bearer_auth("test").send()\n')
            content = content.replace(".send()", '.bearer_auth("test").send()')
            
            # Revert the oneshot channel sends (usually shutdown.send(()))
            content = content.replace('shutdown.bearer_auth("test").send(())', 'shutdown.send(())')
            
            with open(path, 'w') as f:
                f.write(content)
